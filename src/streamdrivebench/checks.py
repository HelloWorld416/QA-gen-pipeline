"""Explicit structured contracts and pure temporal/causal rules."""
import math
from .common import obj_schema, STRING, INTEGER, BOOL, INTS, STRINGS


def array(item):
    return {'type': 'array', 'items': item}

EVENT = obj_schema(dict(actor=STRING, behavior=STRING, before_state=STRING,
    after_state=STRING, start=INTEGER, end=INTEGER, approximate_transition=INTEGER,
    evidence_frames=INTS, uncertainty=STRING))
DISCOVERY = obj_schema({'events': array(EVENT), 'coverage_notes': STRING})
BOUNDARY = obj_schema(dict(valid=BOOL, actor=STRING, behavior=STRING,
    before_state=STRING, after_state=STRING, first_observable=INTEGER,
    last_absent=INTEGER, evidence_frames=INTS, visibility_clear=BOOL,
    difficulty_factors=STRINGS, reason=STRING))
MERGE = obj_schema(dict(same_actor=BOOL, same_occurrence=BOOL,
    same_atomic_transition=BOOL, reason=STRING))
TARGET = obj_schema(dict(identifiable=BOOL, kind={'type':'string','enum':['ego','road_user','scene']},
    point_milli=INTS, description_now=STRING, reason=STRING))
QUESTION = obj_schema(dict(question=STRING, choices=STRINGS, target_description=STRING,
    predicate=STRING, answer_slot=STRING, rationale_from_prefix=STRING))
AUTHOR = obj_schema({'questions': array(QUESTION)})
VERIFY = obj_schema(dict(valid=BOOL, answer=STRING, evidence_frames=INTS,
    target_identifiable=BOOL, unambiguous=BOOL, future_presupposition=BOOL,
    history_necessary=BOOL, reason=STRING))
AD_PREFIX = obj_schema(dict(valid=BOOL, target_identifiable=BOOL, unambiguous=BOOL,
    future_presupposition=BOOL, reason=STRING))
INJECTION = obj_schema(dict(valid=BOOL, target_identifiable=BOOL, driving_state_change=BOOL,
    single_atomic_condition=BOOL, condition_already_true=BOOL,
    future_presupposition=BOOL, reason=STRING))
MONITOR = obj_schema(dict(valid=BOOL, target_consistent=BOOL, visibility_sufficient=BOOL,
    first_observable=INTEGER, last_absent=INTEGER, answer=STRING,
    evidence_frames=INTS, event_absent_throughout=BOOL, reason=STRING))
LINK = obj_schema(dict(answer_supported=BOOL, options_cover_observation=BOOL,
    same_actor=BOOL, same_behavior_episode=BOOL,
    same_atomic_transition=BOOL, semantic_duplicate_indices=INTS,
    relation=STRING, reason=STRING))

DISCOVER_PROMPT = '''You annotate driving events from a forward-facing front-facing dashcam video. Find up to four distinct, clear behavior episodes with an observable atomic state transition in ownership_seconds (inclusive). Frames outside ownership provide context only. Describe what is actually visible: ego motion, road-user movement, stopping/starting, turning, crossing, signal states. No inferred intent, numerical speed, invisible driver decisions or mere camera exit. Give absolute integer frame seconds, visual evidence and uncertainties. State change must have before and after evidence. Repeated behavior at different times is a different occurrence. Empty events is correct if no clear event. This is offline discovery, NOT QA generation. Sparse samples cannot establish precise onset; approximate_transition is only a proposal. start/end delimit this behavior episode, not entire actor visibility.'''
REFINE_PROMPT = '''Independently verify the proposed driving behavior using dense images. Do not trust the proposed description. Identify the first observable second at which the specified atomic transition is satisfied and the immediately preceding observed second where it is absent. Require those two adjacent seconds to be supplied and visually clear; otherwise valid=false. Distinguish onset from ongoing state, ego camera motion from actor behavior, actual motion from perspective changes. Rewrite actor/behavior/states to supported descriptions. evidence_frames must be supplied. Do not claim physical subsecond precision or intent. Reject uncertain visibility or ambiguous identity. Difficulty factors are descriptive hypotheses, not proven model difficulty.'''
MERGE_PROMPT = '''Determine visually whether these two offline annotations concern the same actor, same continuous behavior occurrence, and same atomic transition. Same actor at another time is insufficient. Distinct onset/completion are distinct transitions of one event. Only confirm with images; ambiguous comparisons must be false. Do not merge based solely on language similarity.'''
GROUND_PROMPT = '''This is OFFLINE actor correspondence, not QA authorship. Find the annotated event actor in the supplied query_second image, using other images only to track identity. If it is not visible or uniquely identifiable at query_second, identifiable=false. Return kind, point_milli=[x,y] normalized to 0..1000 marking its center in the QUERY image, and a description of appearance at that moment only. For the camera vehicle use kind=ego and an empty point. Never move the point into a later frame. Do not substitute a nearby actor.'''
SANITIZE_PROMPT = '''You see ONLY a causal video prefix and a point marking a proposed target in the LAST image. You do not know any future event. Independently identify the visible target at that point; if ambiguous or not visible, identifiable=false. Return the same kind and point plus a short description of its CURRENT appearance and location, sufficient for online reference. Do not include future actions, intent, assumed identity, or guesses. Do not infer the purpose of the selection. For kind=ego the target is the camera vehicle.'''
AUTHOR_PROMPT = '''You write online driving QA from ONLY the provided causal video prefix. You have NO information about what happens next. Return zero, one or two useful, semantically distinct questions; no paraphrase duplicates. Use observable target descriptors that identify the actor NOW. When a target is supplied, EVERY question must concern that designated actor; do not substitute a more salient nearby actor. The target descriptor was independently grounded using only the causal prefix and conveys no future action. No inferred future outcome or intention in question wording. Never use a future label as a presupposition. Do not include answers in your output.
HGI: ask about a CURRENT state or identity that genuinely needs historical visual evidence, not a simple fact from the latest frame or arbitrary time lookup. Give 2-4 plausible choices, no SILENT.
AD: ask which observable target behavior/state will be seen at query_second+horizon_seconds. Give 2-4 plausible alternative outcomes without presupposing the actual outcome. No SILENT.
ETP: ask the model to stay silent until ONE observable, atomic condition becomes true for a currently identifiable target, then answer a concrete discriminative question. 2-4 choices plus exactly one SILENT. Condition must be false at injection and plausibly may or may not occur during monitoring. Avoid compound and/or predicates, camera exit, vague soon, hidden physical attributes. Use a specific predicate and answer_slot. Do not force an event that the prefix does not support. Potential negatives and positives use this identical authoring process.
Prefer relevant dynamic interactions, but do not invent movement in static scenes. rationale_from_prefix must cite only supplied observations.'''
VERIFY_PROMPT = '''Independently answer and audit this question using ONLY supplied images. You are NOT given an author answer. valid requires identifiable target, unambiguous choices and visual evidence. answer must exactly match a choice; otherwise mark invalid and answer empty. Cite only supplied evidence_frames. Reject future-presupposing or answer-revealing wording. In HGI full-history mode determine whether history is genuinely required to resolve CURRENT state/identity. In current-only ablation, answer only if the single current frame is sufficient, do not guess. In AD prefix mode do NOT guess the future: assess wording, target, alternatives, and answer empty. In AD future-label mode use the future frames to establish the answer; uncertainty means invalid. Distinguish visible behavior from intent or unobservable metric speed.'''
AD_PREFIX_PROMPT = '''You audit the CAUSAL VALIDITY of a prediction question, NOT its future answer. Supplied images end at query_second; the answer will be independently labeled from future video elsewhere. valid=true when the target is identifiable now, the requested horizon/state is clear, choices are meaningful alternatives, and wording does not presuppose or reveal the actual future outcome. Inability to know the future answer is EXPECTED and MUST NOT be a rejection reason. Do not answer or guess the future. unambiguous refers to target/question meaning, not certainty about the future. Reject only actual flaws in the question or its causal grounding.'''
ETP_AUTHOR_PROMPT = AUTHOR_PROMPT + '''\nETP specialization: prioritize observable DRIVING STATE changes of the designated target: stopping, starting, turn onset/completion, lane entry/change, crossing onset, or a traffic signal changing state. Passing a landmark, another object ceasing to occlude the target, or camera exit is NOT a target driving-state change. Where naturally supported by the current prefix, propose two complementary single-condition hypotheses (for example, a moving ego could stop or begin a turn; author separate questions, never a compound OR condition). Do not require evidence that either hypothesis WILL occur. It is acceptable for a condition to stay absent in the monitoring window; do not invent future knowledge. Do not force questions if no eligible condition can be grounded. Each question must explicitly specify its observable trigger and offer discriminative response choices plus SILENT.'''
INJECTION_PROMPT = '''Audit this ETP monitor question from ONLY the causal prefix ending at query_second, the injection instant. You do not know whether any future event will occur.
Conditional monitoring semantics: "stay silent until X", "when X happens", "when it does", and "at that moment" mean IF X is observed after injection, answer then; OTHERWISE remain SILENT throughout the window. Such conditional instructions do NOT assert that X will happen. Set future_presupposition=false for this wording alone. Reject a genuine assertion of an unobserved future outcome or answer leakage, such as stating that a vehicle WILL stop or WILL turn right as a fact. Judge the full question, not isolated words like when/until.
Current-state semantics: condition_already_true refers to the predicate being satisfied NOW at query_second, established from the latest available temporal evidence. Do not set it true merely because the event occurred earlier in the prefix. A car that stopped earlier but has visibly resumed motion can be monitored for a NEW stop after injection; condition_already_true=false for that stop predicate. A car still stopped now already satisfies a stop predicate and must be rejected. Likewise a moving car already satisfies a start-moving predicate. A signal that changed earlier can be monitored for its next change if the specified initial state holds now. Require a new post-injection transition from the appropriate current state; do not reuse a historical occurrence. If the current state or identity cannot be established, reject as uncertain instead of guessing.
driving_state_change must mean the designated target's own visible motion/state changes (stop/start/turn/lane/crossing/signal). Landmark passing, occlusion clearing alone and camera exit are not eligible changes. The target must be identifiable now and the condition a precise observable single atomic predicate currently unmet. Reject ambiguity, independent compound and/or conditions, an already-satisfied current condition, or genuine future-outcome presupposition. valid=true only when all these requirements hold. Do not waive any other quality criterion because conditional wording is allowed. Explain the CURRENT state and any actual flaw using only supplied evidence.'''
MONITOR_PROMPT = '''Independently monitor every supplied second from injection through deadline. No proposed outcome or trigger is supplied. Track the specified target, test the atomic predicate, and find its FIRST observable satisfaction. Before trigger answer is SILENT; at trigger answer the concrete question with an exact non-SILENT choice. Return first_observable and preceding last_absent. If condition is absent throughout, first_observable=-1, last_absent=deadline, answer=SILENT and event_absent_throughout=true. A disappearance, unresolved occlusion, uncertain identity or insufficient evidence is NOT a negative: valid=false, visibility_sufficient=false. Positive requires absence at injection, then a clear adjacent absent/present boundary. Evidence frames must be supplied. Never shift onset to a later clearer example.'''
LINK_PROMPT = '''Independently CROSS-CHECK THE ANSWER and associate a candidate QA with an offline driving-event annotation using images. Do not trust the supplied label or earlier validation. answer_supported requires the exact answer to be supported at the requested time. options_cover_observation requires the choices to contain the ACTUALLY observed state, not just a closest alternative. Reject inaccurate directional/motion labels, missing correct choices and question/answer inconsistencies, even when the same actor and event are clearly involved. For AD judge the state at query_second+horizon_seconds, not its earlier precursor. State any conflict explicitly with false fields. Temporal proximity alone is insufficient. Check same actor and continuous behavior occurrence. For same_atomic_transition, the actual question predicate/forecast/history evidence must address the annotated state change; an unrelated property of the same car does not qualify. ETP negatives may be related counterfactual monitoring controls; explain relation and do NOT assert same positive transition. Return indices of any semantically duplicate previous questions: wording changes are not new questions, but distinct answer dimensions are; time-shifted same semantics belongs to one probe series. Err on rejection when actor/event linkage is unclear.'''


def windows(n, stride=30):
    for start in range(0, n, stride):
        end = min(n-1, start+stride-1)
        lo, hi = max(0, start-10), min(n-1, end+12)
        frames = sorted(set(range(lo, hi+1, 2)) | {hi, start, end})
        yield start, end, frames


def prefix_frames(t, lookback=30):
    return sorted(set(range(max(0, t-lookback), t+1, 3)) | set(range(max(0,t-5), t+1)))


def causal_payload(stage, frames, t, family, horizon=0, question=None, target=None):
    # Whitelist: do not forward event titles, future actor attributes, boundaries or answers.
    if not frames or max(frames)>t or min(frames)<0:
        raise ValueError('Causal request contains future/invalid frames')
    p=dict(stage=stage, image_frames=frames, query_second=t, family=family,
           horizon_seconds=horizon)
    if target is not None:
        p['target']={k:target[k] for k in ('kind','point_milli','description_now')}
    if question is not None:
        p['question']={k:question[k] for k in ('question','choices','target_description','predicate','answer_slot')}
    return p


def evidence_errors(result, frames):
    if not set(result.get('evidence_frames',[])).issubset(frames):
        return ['evidence_outside_request']
    return []


def monitor_errors(r, t, deadline, choices):
    errors=[]
    if not all(r[k] for k in ('valid','target_consistent','visibility_sufficient')):
        errors.append('monitor_not_valid')
    trigger=r['first_observable']
    if trigger == -1:
        if not r['event_absent_throughout'] or r['answer']!='SILENT' or r['last_absent']!=deadline:
            errors.append('inconsistent_negative')
    else:
        if not t < trigger <= deadline or r['last_absent'] != trigger-1:
            errors.append('invalid_first_boundary')
        if r['event_absent_throughout'] or r['answer']=='SILENT' or r['answer'] not in choices:
            errors.append('inconsistent_positive')
        if not {trigger-1,trigger}.issubset(r['evidence_frames']):
            errors.append('boundary_evidence_missing')
    return errors


def question_errors(q, family):
    errors=[]
    choices=q['choices']
    if not q['question'].strip() or not 2<=len(choices)<=5 or len(set(choices))!=len(choices):
        errors.append('invalid_question_choices')
    if choices.count('SILENT') != (1 if family=='ETP' else 0):
        errors.append('invalid_silence_option')
    return errors


def choose_events(events, limit):
    ordered=sorted(events,key=lambda e:e['transition_second'])
    if not limit or len(ordered)<=limit:return ordered
    # Time-spread pilot coverage; never silently truncate full production runs.
    indices=sorted({round(i*(len(ordered)-1)/max(1,limit-1)) for i in range(limit)})
    return [ordered[i] for i in indices]


def association_errors(link):
    errors=[]
    if not link['same_actor'] or not link['same_behavior_episode']:
        errors.append('wrong_event_or_actor')
    if not link['answer_supported'] or not link['options_cover_observation']:
        errors.append('association_label_conflict')
    return errors


def model_input_key(family, second, question):
    """Same visible task across event aliases; keep different times/targets distinct."""
    normalize = lambda text: ' '.join(text.split())
    return (family, second, normalize(question['question']),
            tuple(sorted(normalize(c) for c in question['choices'])),
            normalize(question.get('target_description', '')),
            normalize(question.get('predicate', '')))
