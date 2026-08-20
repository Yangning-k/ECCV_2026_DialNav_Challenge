from parser import parse_args
import torch
import torch.nn.functional as F
from holistic_utils.distributed import init_distributed, is_default_gpu
from holistic_utils.misc import set_random_seed
from holistic_utils.data_utils import construct_instrs, construct_instrs_universal
from holistic_models.ScaleVLN.ScaleVLN import ScaleVLNModel
from holistic_models.DST.DST import DST
from ModularNavigator import ModularNavigator
from ModularGuide import ModularGuide
from holistic_models.ConfidenceThresholding import (
    ConfidenceThresholdingWtaModule, CappedConfidenceWtaModule)
from holistic_models.LANA.LANA import LANA
from holistic_models.GCNLoc.GCNLoc import GCNLocModel
from holistic_models.GTL.GTL import GraphVlnAgentModel
import time
import numpy as np
from evaluator import Evaluator
import copy
import os
import json
from transformers import logging
import sys
logging.set_verbosity_error()

def get_tokenizer():
    from transformers import AutoTokenizer
    cfg_name = os.environ["DIALNAV_BERT_TOKENIZER_DIR"]
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def load_instruction_data(args, target_envs, tokenizer):
    env_instructions = {}
    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"Invalid split: {split}")
        
        annotation_paths = annotation_paths.split(",")
        instruction_data = construct_instrs(annotation_paths, tokenizer, args.max_instr_len)
        print(f"Loaded instruction data for split: {split} ({annotation_paths}) with length: {len(instruction_data)}")
        env_instructions[split] = instruction_data
    return env_instructions

def load_instruction_data_universal(args, target_envs, tokenizer):
    env_instructions = {}

    if args.benchmark in ['cvdn', 'dialnav']:
        prefix = "target : "
    else:
        prefix = ""

    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"Invalid split: {split}")

        annotation_paths = annotation_paths.split(",")
        instruction_data = construct_instrs_universal(annotation_paths, tokenizer, args.max_instr_len, prefix)
        print(f"Loaded instruction data for split: {split} ({annotation_paths}) with length: {len(instruction_data)}")
        env_instructions[split] = instruction_data
    return env_instructions

def dialog_setup(benchmark):
    import os
    override = os.environ.get("UPDATE_ANSWER_BEHIND")
    if override is not None:
        return override == "1"
    if benchmark == 'cvdn':
        return True
    elif benchmark == 'dialnav':
        return True
    else:
        return False


def _follow_waypoint_plan(graph, scan, cur, gpath):
    """Advance along a waypoint plan.

    If the agent is on the plan, move to the next waypoint; if it reached the
    end, stop; otherwise step toward the nearest waypoint on the scene graph.
    Returns (next_vp, force_stop).
    """
    if not gpath:
        return None, False
    if cur == gpath[-1]:
        return None, True
    if cur in gpath:
        idx = gpath.index(cur)
        if idx + 1 < len(gpath):
            return gpath[idx + 1], False
        return None, True
    best_path = None
    best_dist = None
    try:
        for node in gpath:
            dist = graph["shortest_distances"][scan][cur][node]
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_path = graph["shortest_paths"][scan][cur][node]
    except Exception:
        return None, False
    if best_path is not None and len(best_path) > 1:
        return best_path[1], False
    return None, False


def _format_answer_for_nav(answer):
    """Return the Guide's natural-language answer unchanged for the Navigator."""
    return answer


def _ground_answer_goal_candidates(ans_model, graph, scan, question_viewpoint, answer, k, alpha):
    """Return ranked candidate goal viewpoints with their shortest paths.

    This is the compliant, language-derived candidate set used by the
    path-language reranker.
    """
    if ans_model is None:
        return []
    loc_agent = getattr(ans_model, "agent", None)
    if loc_agent is None:
        return []
    try:
        ans_model.localize([scan], [answer])
    except Exception:
        return []
    logits = getattr(loc_agent, "last_loc_logits", None)
    nodes = getattr(loc_agent, "last_loc_nodes", None)
    if logits is None or nodes is None or not nodes:
        return []
    probs = torch.softmax(logits, dim=1)
    topk = torch.topk(probs[0], min(k, probs[0].shape[0]))
    dist_map = graph["shortest_distances"][scan][question_viewpoint]
    candidates = []
    for idx, val in zip(topk.indices, topk.values):
        c = nodes[0][int(idx)]
        p = float(val)
        if c == question_viewpoint:
            continue
        d = dist_map.get(c, -1)
        score = d + alpha * p
        try:
            path = list(graph["shortest_paths"][scan][question_viewpoint][c])
        except Exception:
            continue
        candidates.append((c, p, score, path))
    if not candidates and len(topk.indices) > 0:
        c = nodes[0][int(topk.indices[0])]
        try:
            path = list(graph["shortest_paths"][scan][question_viewpoint][c])
        except Exception:
            return []
        candidates.append((c, float(topk.values[0]), 0.0, path))
    return candidates


def _ground_answer_goal_candidates_multi(ans_model, graph, scan, question_viewpoint,
                                         texts, k, alpha):
    """Union candidate goals from several answer-text variants.

    The answer localizer is sensitive to how much of the Guide's long answer is
    fed to it.  Unioning the top-k candidates from full/tail/last-sentence/qa
    variants can recover a true target that any single formatting misses.
    """
    merged = {}
    for text in texts:
        for cand in _ground_answer_goal_candidates(
                ans_model, graph, scan, question_viewpoint, text, k, alpha):
            c, p, score, path = cand
            old = merged.get(c)
            if old is None or p > old[1]:
                merged[c] = cand
    return sorted(merged.values(), key=lambda x: -x[1])[:k]


def _ground_gtl_candidates(loc_model, graph, scan, question_viewpoint, text, k, alpha):
    """Return ranked goal candidates from the Navigator-side GTL localizer.

    This is a complementary candidate source to the dedicated answer-localizers:
    the same question/answer text is matched against all scene viewpoints with
    the released localization model.  Keeping its top-k nodes in the pool gives
    the path-language reranker more chances to recover the true target without
    using the disclosed ground-truth target directly.
    """
    if loc_model is None:
        return []
    agent = getattr(loc_model, "agent", None)
    if agent is None:
        return []
    try:
        loc_model.localize([scan], [text])
    except Exception:
        return []
    logits = getattr(agent, "last_loc_logits", None)
    nodes = getattr(agent, "last_loc_nodes", None)
    if logits is None or nodes is None or not nodes:
        return []
    probs = torch.softmax(logits, dim=1)
    topk = torch.topk(probs[0], min(k, probs[0].shape[0]))
    dist_map = graph["shortest_distances"][scan][question_viewpoint]
    candidates = []
    for idx, val in zip(topk.indices, topk.values):
        c = nodes[0][int(idx)]
        p = float(val)
        if c == question_viewpoint:
            continue
        d = dist_map.get(c, -1)
        try:
            path = list(graph["shortest_paths"][scan][question_viewpoint][c])
        except Exception:
            continue
        candidates.append((c, p, d + alpha * p, path))
    if not candidates and len(topk.indices) > 0:
        c = nodes[0][int(topk.indices[0])]
        try:
            path = list(graph["shortest_paths"][scan][question_viewpoint][c])
        except Exception:
            return []
        candidates.append((c, float(topk.values[0]), 0.0, path))
    return candidates


def _score_paths_with_answer(answer_model, scan, paths, answer, batch_size=1,
                             question=None, score_text_mode="answer"):
    """Score candidate paths by the LANA answer model's token likelihood.

    Teacher-forcing is used over the complete answer sequence in a single
    forward pass instead of re-running the decoder token by token.
    """
    if answer_model is None or not paths:
        return []
    agent = getattr(answer_model, "agent", None)
    env = getattr(answer_model, "language_env", None)
    if agent is None or env is None:
        return []
    if score_text_mode == "qa" and question:
        text = (question + " " + answer).strip()
    elif score_text_mode == "tail":
        text = " ".join(answer.split()[-20:])
    elif score_text_mode == "last":
        parts = [p.strip() for p in answer.replace(" .", ".").split(". ")]
        parts = [p for p in parts if p]
        text = parts[-1] if parts else answer
    else:
        text = answer
    try:
        token_ids = agent.tokenizer.encode(text, max_length=200, truncation=True)
    except TypeError:
        # The CLIP SimpleTokenizer used by the LANA/CLIP16 checkpoint has a
        # minimal ``encode(text)`` signature and no truncation options.
        token_ids = agent.tokenizer.encode(text)[:200]
    scores = []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        env.reset([scan] * len(chunk),
                  [p[0] for p in chunk],
                  [3.14] * len(chunk),
                  [])
        t_hist, t_act, hist_lens, action_lens, _ = agent.get_history_and_actions_for_speaker(env, chunk)
        bs = len(chunk)
        hist_embeds = [agent.vln_bert('history').expand(bs, -1, -1)]
        action_embeds = []
        for t, action_input in enumerate(t_act):
            t_hist_inputs = t_hist[t]
            hist_embeds.append(agent.vln_bert(**t_hist_inputs))
            action_embeds.append(agent.vln_bert(**action_input).unsqueeze(1))
        max_len = len(token_ids)
        words = torch.zeros(bs, max_len, dtype=torch.long, device='cuda')
        for j, ids in enumerate([token_ids] * bs):
            words[j, :len(ids)] = torch.tensor(ids, dtype=torch.long, device='cuda')
        future_mask = agent.make_future_mask(words.shape[1], hist_embeds[0].dtype, words.device)
        caption_lengths = (words != 0).sum(-1)
        ones = torch.ones_like(words)
        caption_mask = caption_lengths.unsqueeze(1) < ones.cumsum(dim=1)
        language_inputs = {
            'mode': 'language',
            'txt_ids': words,
            'txt_masks': caption_mask,
            'future_mask': future_mask,
        }
        txt_embeds = agent.vln_bert(**language_inputs)
        caption_input = {
            'mode': 'visual',
            'hist_embeds': hist_embeds,
            'txt_embeds': txt_embeds,
            'txt_masks': caption_mask,
            'hist_lens': hist_lens,
            'action_embeds': action_embeds,
            'action_lens': action_lens,
            'is_train_caption': True,
            'future_mask': future_mask,
        }
        logits = agent.vln_bert(**caption_input)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = words[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=agent.pad_token_id,
            reduction='none',
        ).view(bs, -1)
        valid = (shift_labels != agent.pad_token_id)
        scores.extend(
            (-ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1))
            .detach().cpu().tolist()
        )
    return scores


def _load_path_rerank_weights(answer_model, ckpt_path):
    """Load a contrastive path-answer reranker checkpoint.

    The training script stores the raw ``vln_bert`` / ``critic`` state dicts,
    which is deliberately different from ``LanaSpeaker.load``'s nested
    checkpoint format.  Loading here is explicit and non-strict so it can
    tolerate minor key differences while still replacing the scoring model.
    """
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=True)
    if "vln_bert" not in ckpt:
        raise ValueError(f"Reranker checkpoint {ckpt_path} has no vln_bert state")
    answer_model.agent.vln_bert.load_state_dict(ckpt["vln_bert"], strict=False)
    if "critic" in ckpt:
        answer_model.agent.critic.load_state_dict(ckpt["critic"], strict=False)
    answer_model.agent.vln_bert.eval()
    answer_model.agent.critic.eval()
    return answer_model


def dialNav(navigator, 
            guide,
            mode,
            max_action_len=50, update_answer_behind=False, loc_conf_threshold=0.0,
            answer_grounding_model=None, answer_grounding_model2=None,
            path_rerank_model=None):
    # At each clarification turn the Navigator asks a question, the Guide
    # answers in natural language, and the answer (with the question) is
    # grounded into candidate destinations by the answer/QA localizers. A
    # path-language reranker then scores the candidate shortest paths and the
    # best one is executed as a waypoint plan. Everything is derived from the
    # natural-language dialog and scene connectivity only.
    navigator.set_next_batch()
    obs = navigator.get_obs()
    batch_size = len(obs)


    traj = [{
            'scan': ob['scan'],
            'start_pano': ob['viewpoint'],
            'end_panos': ob['end_panos'],
            'target': ob['instruction'],
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'navigation_detail': []
        } for idx, ob in enumerate(obs)]
    navigator.initialize_nav(obs)

    ask = np.array([False] * batch_size)


    question_seen_path = None
    answer_seen_path = None
    waypoint_plans = {}
    graph_info = {
        "shortest_distances": guide.shortest_distances,
        "shortest_paths": guide.shortest_paths,
    }
    for step in range(max_action_len):
        if step % 5 == 0:
            print(f"[MEM] step={step} allocated={torch.cuda.memory_allocated()/1e9:.2f}G "
                  f"reserved={torch.cuda.memory_reserved()/1e9:.2f}G", flush=True)
        waypoint_next = {}
        waypoint_end = {}
        for i in range(batch_size):
            if i in waypoint_plans and waypoint_plans[i] and not ended[i]:
                nv, stop = _follow_waypoint_plan(
                    graph_info, obs[i]['scan'], obs[i]['viewpoint'], waypoint_plans[i])
                if nv is not None:
                    waypoint_next[i] = nv
                if stop:
                    waypoint_end[i] = True
        next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)
        for i, nv in waypoint_next.items():
            next_vp_ids[i] = nv
            ended[i] = False
        for i in waypoint_end:
            ended[i] = True
            next_vp_ids[i] = None
        next_vp_ids_before_dialog = copy.deepcopy(next_vp_ids)
        ended_before_this_step = copy.deepcopy(ended)

        ## details
        nav_probs_cache = nav_probs.clone()
        c = torch.distributions.Categorical(nav_probs)
        c_cache = torch.distributions.Categorical(nav_probs_cache)

        if mode == 'navonly':
            ask = np.array([False] * batch_size)
        else:
            ask = navigator.wta(step, nav_probs, nav_outs)
            
        to_ask_indices = [index for index, value in enumerate(ask) if value and not ended[index]]
        need_dialog = len(to_ask_indices) > 0
        if need_dialog:
            print(f"[Step {step}] to_ask_indices: {to_ask_indices}")
        scanIds = [obs[i]['scan'] for i in range(batch_size)]
        viewpoints = [obs[i]['viewpoint'] for i in range(batch_size)]
        # The Guide knows the destination by task design; its target list is
        # used only to produce the natural-language answer.
        guide_goals = [obs[i]['end_panos'] for i in range(batch_size)]

        if need_dialog:
            questions, question_seen_path = navigator.ask(scanIds, viewpoints)
            localized_viewpoints = guide.localize(scanIds, questions)

            # Localization-confidence gating: when the GTL localization is
            # uncertain, do NOT feed the answer back into the navigator (it
            # would likely misdirect navigation). The question still counts
            # toward DTC; only the instruction update is skipped.
            loc_conf = None
            loc_agent = getattr(guide.localization_model, "agent", None)
            if loc_agent is not None:
                loc_conf = getattr(loc_agent, "last_loc_conf", None)
            if loc_conf_threshold > 0 and loc_conf is not None:
                update_indices = [
                    i for i in to_ask_indices if loc_conf[i] >= loc_conf_threshold
                ]
                skipped = [
                    i for i in to_ask_indices if loc_conf[i] < loc_conf_threshold
                ]
                if skipped:
                    print(f"[LocGate] threshold={loc_conf_threshold} "
                          f"skipped {len(skipped)} uncertain updates: {skipped} "
                          f"confs={[round(loc_conf[i], 3) for i in skipped]}")
            else:
                update_indices = to_ask_indices
            
            paths = [guide._choose_path(scanId, viewpoint, goal)
                     for scanId, viewpoint, goal
                     in zip(scanIds, localized_viewpoints, guide_goals)]

            answers, answer_seen_path = guide.answer(scanIds, localized_viewpoints, paths)
            answers_for_update = [_format_answer_for_nav(a) for a in answers]
            navigator.update_instruction(update_indices, questions, answers_for_update, append_behind=update_answer_behind)
            # Candidate generation and path-language reranking. The answer
            # localizer is applied to several text variants (tail, last
            # sentence, question+answer), the QA localizer is applied to the
            # answer, and the released localizer contributes complementary
            # candidates. Every candidate is converted to the graph shortest
            # path from the localized viewpoint; the reranker scores them and
            # the best path is stored as the waypoint plan.
            rerank_k = int(os.environ.get("RERANK_K", "80"))
            rerank_alpha = float(os.environ.get("RERANK_ALPHA", "5"))
            rerank_batch = int(os.environ.get("RERANK_BATCH", "2"))
            rerank_texts = os.environ.get("RERANK_TEXTS", "tail,last,qa")
            loc_cand_k = int(os.environ.get("LOC_CAND_K", "20"))
            for i in to_ask_indices:
                text_variants = []
                for spec in rerank_texts.split(","):
                    spec = spec.strip()
                    if spec == "answer":
                        text_variants.append(answers[i])
                    elif spec == "tail":
                        text_variants.append(" ".join(answers[i].split()[-20:]))
                    elif spec == "last":
                        parts = [p.strip() for p in answers[i].replace(" .", ".").split(". ")]
                        parts = [p for p in parts if p]
                        text_variants.append(parts[-1] if parts else answers[i])
                    elif spec == "qa":
                        text_variants.append((questions[i] + " " + answers[i]).strip())
                cands = _ground_answer_goal_candidates_multi(
                    answer_grounding_model, graph_info, scanIds[i],
                    localized_viewpoints[i], text_variants,
                    rerank_k, rerank_alpha)[:rerank_k]
                if answer_grounding_model2 is not None:
                    cands2 = _ground_answer_goal_candidates(
                        answer_grounding_model2, graph_info, scanIds[i],
                        localized_viewpoints[i], answers[i],
                        rerank_k, rerank_alpha)
                    seen = {c[0] for c in cands}
                    cands = cands + [c for c in cands2 if c[0] not in seen]
                    cands = cands[:max(rerank_k, 2 * rerank_k)]
                if loc_cand_k > 0:
                    loc_cands = _ground_gtl_candidates(
                        guide.localization_model, graph_info, scanIds[i],
                        localized_viewpoints[i], answers[i],
                        loc_cand_k, rerank_alpha)
                    seen = {c[0] for c in cands}
                    cands = cands + [c for c in loc_cands if c[0] not in seen]
                    cands = cands[:max(rerank_k, loc_cand_k, 2 * rerank_k)]
                if cands:
                    rerank_model = path_rerank_model or guide.answer_model
                    path_scores = _score_paths_with_answer(
                        rerank_model, scanIds[i],
                        [c[3] for c in cands], answers[i],
                        batch_size=rerank_batch,
                        question=questions[i],
                        score_text_mode="answer")
                    best_idx = max(range(len(path_scores)),
                                   key=lambda j: path_scores[j])
                    gv, conf, _, path = cands[best_idx]
                    conf = float(conf)
                    waypoint_plans[i] = list(path)
                else:
                    gv = None
                    conf = None
                    waypoint_plans[i] = None
                print(f"[PlanSelect] instr={obs[i]['instr_id']} "
                      f"goal={None if gv is None else gv[:8]} "
                      f"conf={None if conf is None else round(conf,3)} "
                      f"path_len={None if not waypoint_plans[i] else len(waypoint_plans[i])}")

            next_vp_ids, ended, nav_probs, instrucion_for_this_nav, nav_outs = navigator.get_next_action(step, obs)

        obs, paths = navigator.navigate(next_vp_ids, obs, ended, traj)
        just_ended = ended & ~ended_before_this_step
        if waypoint_end:
            # The DST stop-node logic re-anchors the recorded path to the
            # model's preferred stop node even when waypoint execution forced
            # the stop at the selected goal. The simulator is actually at the
            # goal, so truncate the recorded path to the real final position.
            for i in waypoint_end:
                goal_vp = waypoint_plans[i][-1]
                flat = [v for st in traj[i]['path'] for v in (st if isinstance(st, list) else [st])]
                if flat and flat[-1] != goal_vp and goal_vp in flat:
                    last_idx = len(flat) - 1 - flat[::-1].index(goal_vp)
                    traj[i]['path'] = [[v] for v in flat[:last_idx + 1]]

        ### update trajectory log
        c = torch.distributions.Categorical(nav_probs)
        for i in range(batch_size):
            if ended[i] and not just_ended[i]:
                continue
            
            navigation_detail_item = {
                'nav_idx': step,
                'ask': False,
                'instruction': instrucion_for_this_nav[i],
                'gt_viewpoint': viewpoints[i],
                'next_vp_ids': next_vp_ids[i],
                'ended': ended[i],
                # 'nav_probs': nav_probs[i],
                'entropy': c.entropy()[i].item(),
            }
            if i in to_ask_indices:
                navigation_detail_item['ask'] = True
                navigation_detail_item['question'] = questions[i]
                navigation_detail_item['localized_viewpoint'] = localized_viewpoints[i]
                navigation_detail_item['answer'] = answers[i]
                # navigation_detail_item['gt_viewpoint'] = viewpoints[i]
                if question_seen_path:
                    navigation_detail_item['question_seen_path'] = question_seen_path[i]
                if answer_seen_path:
                    navigation_detail_item['answer_seen_path'] = answer_seen_path[i]
                navigation_detail_item['vp_before_dialog'] = next_vp_ids_before_dialog[i]
                navigation_detail_item['entropy_before_dialog'] = c_cache.entropy()[i].item()
                navigation_detail_item['entropy_diff'] = navigation_detail_item['entropy'] - navigation_detail_item['entropy_before_dialog']
            traj[i]['navigation_detail'].append(navigation_detail_item)

            ## already processed in make_equiv_action
            # if not just_ended[i]:
            #     traj[i]['path'].append(paths[i])

        if all(ended):
            break

    return traj

def run(navigator, guide, max_action_len, mode, env_name, output_file, benchmark,
        loc_conf_threshold=0.0, answer_grounding_model=None,
        answer_grounding_model2=None, path_rerank_model=None):
    print("evaluating on env: ", env_name)
    start_time = time.time()

    # Set the target environment
    navigator.set_target_env(env_name)

    # Reset the data index to beginning of epoch. 
    navigator.reset_epoch()
    results = {}

    index = 1
    finished = False
    while not finished:
        print(f"Processing data {index}")
        index += 1
        # if index > 2:
        #     finished = True
        
        trajectories = dialNav(
            navigator, 
            guide, 
            max_action_len=max_action_len, 
            mode=mode,
            update_answer_behind=dialog_setup(benchmark),
            loc_conf_threshold=loc_conf_threshold,
            answer_grounding_model=answer_grounding_model,
            answer_grounding_model2=answer_grounding_model2,
            path_rerank_model=path_rerank_model,
        )
        for traj in trajectories:
            if traj['instr_id'] in results:
                finished = True
            if not finished:
                results[traj['instr_id']] = traj
            
        ### make output in list
        output = [{'instr_id': k, **v} for k, v in results.items()]
        ## save output to json
        with open(output_file, "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)
    print("finished all trajectories", len(results))
    print("time taken: ", time.time() - start_time, "seconds")

    
    return output


def flatten_path(path):
    flat_path = []
    for step in path:
        if isinstance(step, list):
            flat_path.extend(flatten_path(step))
        else:
            flat_path.append(step)
    return flat_path


def make_submit_output(output):
    submit_output = []
    for item in output:
        submit_item = {k: v for k, v in item.items() if k != 'navigation_detail'}
        submit_item['path'] = flatten_path(item.get('path', []))
        ## remove start_pano, end_panos, nav_error, and gt_path from submit output
        submit_item.pop('start_pano', None)
        submit_item.pop('end_panos', None)
        submit_item.pop('nav_error', None)
        submit_item.pop('gt_path', None)

        dialog = []
        for detail in item.get('navigation_detail', []):
            if detail.get('ask'):
                dialog.append({
                    'nav_idx': detail.get('nav_idx'),
                    'question': detail.get('question'),
                    'answer': detail.get('answer'),
                    'localized_viewpoint': detail.get('localized_viewpoint'),
                    'viewpoint': detail.get('gt_viewpoint'),
                })
        submit_item['dialog'] = dialog
        submit_output.append(submit_item)

    return submit_output
            
def setWta(wta_mode, navigation_model=None):
    if wta_mode.startswith('ct'):
        parts = wta_mode.split('_')
        threshold = float(parts[1])
        cap = int(parts[3]) if len(parts) > 3 and parts[2] == 'cap' else None
        min_step = int(parts[5]) if len(parts) > 5 and parts[4] == 'min' else 0
        if cap is not None:
            print(f"Setting wta to confidence thresholding with threshold {threshold}, cap {cap}, min_step {min_step}")
            return CappedConfidenceWtaModule(threshold=threshold, cap=cap, min_step=min_step)
        print("Setting wta to confidence thresholding with threshold", threshold)
        return ConfidenceThresholdingWtaModule(threshold=threshold)
    raise ValueError(f"Unsupported wta_mode: {wta_mode}")
    
def setAgents(args, target_envs, env_instructions, evaluator, scans):

    if args.nav_model == 'ScaleVLN':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modules_path = os.path.join(current_dir, '../../../modules/nav/ScaleVLN/map_nav_src')
        sys.path.insert(0, modules_path)
        ### Initialize Modules
        navigation_model_args = {
            'batch_size': args.batch_size, 
            'basepath': args.basepath, 
            'resume_file': args.nav_resume_file,
            'act_visited_nodes': args.nav_act_visited_nodes,
            'connectivity_dir': args.connectivity_dir,
            'wta_question_threshold': args.nav_wta_question_threshold,
        }
        navigation_model = ScaleVLNModel(args.basepath, navigation_model_args)
        navigation_model.eval()
        navigation_model.set_envs(target_envs, env_instructions)
    elif args.nav_model == 'DST':
        current_dir = os.path.dirname(os.path.abspath(__file__))
        modules_path = os.path.join(current_dir, '../../../modules/nav/DST/map_nav_src')
        sys.path.insert(0, modules_path)
        navigation_model_args = {
            'batch_size': args.batch_size, 
            'basepath': args.basepath, 
            'resume_file': args.nav_resume_file,
            'act_visited_nodes': args.nav_act_visited_nodes,
            'question_weight': args.nav_wta_question_threshold,
        }
        navigation_model = DST(args.basepath, navigation_model_args)
        navigation_model.eval()
        navigation_model.set_envs(target_envs, env_instructions)
    else:
        raise ValueError(f"Invalid navigation model: {args.nav_model}")
    localization_model = None
    question_model = None
    answer_model = None
    wta_model = None
    if args.mode != 'navonly':
        wta_model = setWta(args.wta_mode, navigation_model)
        answer_model = LANA(args.basepath, {
            'scan_list': scans,
            'resume_file': args.ag_resume_file,
            'connectivity_dir': args.connectivity_dir,
            'bpe_path': args.qa_clip_tokenizer_path,
            'max_action_len': args.ag_max_answer_seen_path,
        }, type='ag')

        if args.mode != 'navonly':
            question_model = LANA(args.basepath, {
                'scan_list': scans,
                'resume_file': args.qg_resume_file,
                'connectivity_dir': args.connectivity_dir,
                'bpe_path': args.qa_clip_tokenizer_path,
            }, type='qg')

            if args.loc_model == 'GCN':
                localization_model = GCNLocModel(args.basepath, {
                    'eval_ckpt': args.loc_resume_file,
                    'panofeat_dir': args.loc_node_feats_dir,
                    'geodistance_file': args.loc_geodistance_nodes_path,
                    'connect_dir': args.connectivity_dir+"/",
                    'embedding_dir': args.loc_embedding_dir,
                    'bert_enc': args.loc_bert_enc,
                })
            elif args.loc_model == 'GTL':
                localization_model = GraphVlnAgentModel(args.basepath, {
                    'resume_file': args.loc_resume_file,
                    'scan_list': scans,
                })
            else:
                raise ValueError(f"Invalid localization model: {args.loc_model}")

    answer_grounding_model = None
    answer_grounding_model2 = None
    ans_ckpt = os.environ.get("ANS_CKPT", "")
    if ans_ckpt:
        answer_grounding_model = GraphVlnAgentModel(args.basepath, {
            'resume_file': ans_ckpt,
            'scan_list': scans,
        })
    ans_ckpt2 = os.environ.get("ANS_CKPT2", "")
    if ans_ckpt2:
        answer_grounding_model2 = GraphVlnAgentModel(args.basepath, {
            'resume_file': ans_ckpt2,
            'scan_list': scans,
        })

    path_rerank_model = None
    if os.environ.get("RERANK_CKPT"):
        path_rerank_model = LANA(args.basepath, {
            'scan_list': scans,
            'resume_file': args.ag_resume_file,
            'connectivity_dir': args.connectivity_dir,
            'bpe_path': args.qa_clip_tokenizer_path,
            'max_action_len': args.ag_max_answer_seen_path,
        }, type='ag')
        _load_path_rerank_weights(
            path_rerank_model,
            os.environ["RERANK_CKPT"])

    env_infos = {"shortest_distances": evaluator.shortest_distances, "shortest_paths": evaluator.shortest_paths}
    guide_agent = ModularGuide(args, answer_model, localization_model, env_infos)
    navigator_agent = ModularNavigator(args, navigation_model, wta_model, question_model)
    return (navigator_agent, guide_agent, answer_grounding_model,
            answer_grounding_model2, path_rerank_model)



def main():
    print("run parser")
    args = parse_args()
    target_envs = args.env_names.split(",")


    ### make output path
    os.makedirs(args.output_path, exist_ok=True)

    print("MAIN ARGS")
    print("args", args)
    args_log_file = f"{args.output_path}/args.txt"
    with open(args_log_file, "w") as f:
        f.write("--- MAIN ARGS --- \n")
        f.write("Time: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
        for key, value in vars(args).items():
            if isinstance(value, (np.int64, np.float64)):
                value = value.item()
            f.write(f"{key}: {value}\n")
        f.write("\n\n")

    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    else:
        rank = 0


    set_random_seed(args.seed + rank)

    tokenizer = get_tokenizer()
    env_instructions = load_instruction_data_universal(args, target_envs, tokenizer)
    if args.debug and 'val_seen' in env_instructions:
        env_instructions['val_seen'] = env_instructions['val_seen'][:16]
        # print(f"Debug mode enabled: using first 16 samples from val_seen only.")
        # target_envs = ['val_seen']
        env_instructions['val_unseen'] = env_instructions['val_unseen'][:16]
        env_instructions['test'] = env_instructions['test'][:16]

    ## set up evaluator
    scans = list(set([item['scan'] for env_name in target_envs for item in env_instructions[env_name]]))
    evaluator = Evaluator(
        args.connectivity_dir,
        scans,
        success_margin=args.success_margin,
        error_margin=args.error_margin,
    )

    (navigator_agent, guide_agent, answer_grounding_model,
     answer_grounding_model2, path_rerank_model) = setAgents(
        args, target_envs, env_instructions, evaluator, scans)

    with open(args_log_file, "a") as f:
        targets = {
            "navigation": navigator_agent.navigation_model.args,
        }
        if args.mode != 'navonly':
            targets['answer_generation'] = guide_agent.answer_model.args
        if args.mode == 'holistic':
            targets['question_generation'] = navigator_agent.question_generation_model.args
            targets['localization'] = guide_agent.localization_model.args
        
        f.write("--- NAVIGATOR ARGS --- \n")
        for key, value in targets.items():
            f.write(f"{key}: \n")
            for key, value in vars(value).items():
                if isinstance(value, (np.int64, np.float64)):
                    value = value.item()
                f.write(f"{key}: {value}\n")
            f.write("\n\n")


    submit_output = {}
    for env_name in target_envs:
        metrics_acc = {}
        avg_metrics_acc = {}
        output = run(
            navigator_agent, 
            guide_agent, 
            max_action_len=args.max_action_len,
            mode=args.mode,
            env_name=env_name,
            output_file=f"{args.output_path}/{env_name}.json",
            benchmark=args.benchmark,
            loc_conf_threshold=args.loc_conf_threshold,
            answer_grounding_model=answer_grounding_model,
            answer_grounding_model2=answer_grounding_model2,
            path_rerank_model=path_rerank_model,
        )


        for item in output:
            item['nav_error'] = float(evaluator.get_shortest(item['scan'], item['path'][-1][-1], item['end_panos']))
            for detail in item['navigation_detail']:
                if 'localized_viewpoint' in detail:
                    detail['loc_error'] = float(evaluator.get_shortest(item['scan'], detail['gt_viewpoint'], [detail['localized_viewpoint']]))


        ## save output to json
        with open(f"{args.output_path}/{env_name}.json", "w") as f:
            json.dump(output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)

        submit_output[env_name]=make_submit_output(output)

        avg_metrics, metrics = evaluator.eval_metrics(output)
        metrics_acc[env_name] = metrics
        avg_metrics_acc[env_name] = avg_metrics
        avg_metrics_acc[env_name]['Agg'] = f"{','.join([str(round(avg_metrics_acc[env_name][key], 2)) for key in ['sr','oracle_sr','spl','nav_error','steps','dtc','le']])}"

        with open(f"{args.output_path}/avg_metrics_{env_name}.json", "w") as f:
            json.dump({'avg_metrics_acc': avg_metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)
        with open(f"{args.output_path}/metrics_{env_name}.json", "w") as f:
            json.dump({'metrics_acc': metrics_acc[env_name]}, f, default=lambda x: x.item() if isinstance(x, (np.int64, np.float64)) else x)

    with open(f"{args.output_path}/submit.json", "w") as f:
        json.dump(submit_output, f, default=lambda x: x.item() if isinstance(x, (bool, np.bool_)) else x)


if __name__ == '__main__':
    main()
