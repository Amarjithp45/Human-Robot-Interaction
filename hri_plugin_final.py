# ─────────────────────────────────────────────────────────────────────────────
# HRI Python Plugin — Finger Intention Recognition
# Course   : EE653 Human-Robot Interaction
# Simulator: https://hri-sim2.jahanzebgul.com/
# ─────────────────────────────────────────────────────────────────────────────
#
# HRI PROBLEM:
#   How can a human clearly and safely communicate to a robot
#   which operation to perform using only hand gestures?
#
# SYSTEM DESIGN:
#   1. PERCEPTION        — webcam + MediaPipe reads hand finger data
#   2. INTENTION EQUATION— combines finger clarity + stability into a score
#   3. PLANNING          — only acts when score is high enough
#   4. ROBOT ACTION      — triggers the correct saved operation
#   5. FEEDBACK          — debug panel shows live system thinking
#
# INTENTION RECOGNITION EQUATION:
#
#   intention_score = finger_score × stability_score
#
#   finger_score    = how clearly the finger count maps to an operation (0→1)
#   stability_score = how consistently the gesture has been held (0→1)
#
#   The robot only acts when intention_score >= THRESHOLD (0.70)
#
#   This is different from simple if/else finger counting because:
#   - A fleeting gesture gives low stability_score → robot does NOT trigger
#   - Only a clear, sustained gesture gives high score → robot triggers
#   - This models real HRI: deliberate intent, not accidental gestures
#
# GESTURE → OPERATION MAPPING:
#   1 finger  → Operation 1
#   2 fingers → Operation 2
#   3 fingers → Operation 3
#   4 fingers → Operation 4

# ─────────────────────────────────────────────────────────────────────────────
# PLUGIN METADATA
# ─────────────────────────────────────────────────────────────────────────────

PLUGIN_META = {
    "name": "Finger Intention Recognition",
    "description": (
        "Recognises intended robot operation from finger count. "
        "Uses intention_score = finger_score x stability_score. "
        "Robot only triggers when intention is clear and sustained."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Minimum intention score to trigger robot (0.0 to 1.0)
INTENTION_THRESHOLD = 0.70

# How many frames the gesture must be held before score is high enough
REQUIRED_STABLE_FRAMES = 4

# Milliseconds before the same operation can trigger again
COOLDOWN_MS = 2000

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

LAST_TRIGGER_AT  = 0
LAST_TRIGGER_KEY = None
STABILITY        = {}


# ─────────────────────────────────────────────────────────────────────────────
# SETUP — called once when file is uploaded to simulator
# ─────────────────────────────────────────────────────────────────────────────

def setup(payload):
    global LAST_TRIGGER_AT, LAST_TRIGGER_KEY, STABILITY
    LAST_TRIGGER_AT  = 0
    LAST_TRIGGER_KEY = None
    STABILITY        = {}
    return {
        "status": "Finger Intention Plugin ready",
        "available_operations": len(payload.get("saved_operations", [])),
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — safely get a saved operation by index
# ─────────────────────────────────────────────────────────────────────────────

def operation_by_index(frame, index):
    saved = frame.get("saved_operations", [])
    if index is None or index < 0 or index >= len(saved):
        return None
    return saved[index]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — update stability counter
# ─────────────────────────────────────────────────────────────────────────────

def update_stability(key):
    global STABILITY
    STABILITY = {key: STABILITY.get(key, 0) + 1}
    return STABILITY[key]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PERCEPTION
# Read finger count from hand data
# ─────────────────────────────────────────────────────────────────────────────

def read_finger_count(hand):
    """
    Read how many fingers are extended (not counting thumb).
    Uses simulator's pre-computed value when available.
    Falls back to counting from fingerStates if needed.
    """
    runtime = hand.get("fingerCountNoThumb")
    if runtime is not None:
        return int(runtime)
    states = hand.get("fingerStates") or {}
    return sum(
        1 for k in ("indexExtended", "middleExtended", "ringExtended", "pinkyExtended")
        if states.get(k)
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — INTENTION RECOGNITION EQUATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_finger_score(count):
    """
    finger_score: how clearly does this finger count map to an operation?

    Clean counts (1,2,3,4) = full score 1.0
    0 fingers (fist)       = unclear = 0.0
    5 fingers (open palm)  = slightly ambiguous = 0.7
    """
    if count == 1: return 1.0, 0    # 1 finger → operation slot 0
    if count == 2: return 1.0, 1    # 2 fingers → operation slot 1
    if count == 3: return 1.0, 2    # 3 fingers → operation slot 2
    if count == 4: return 1.0, 3    # 4 fingers → operation slot 3
    if count == 5: return 0.7, 3    # open palm → operation slot 3 (less clear)
    return 0.0, None                # fist or no fingers → unclear


def compute_stability_score(stable_frames):
    """
    stability_score: how long has this gesture been held?

    Increases from 0.0 to 1.0 as stable_frames increases.
    Reaches 1.0 when stable_frames >= REQUIRED_STABLE_FRAMES.

    Formula: stability_score = stable_frames / REQUIRED_STABLE_FRAMES
             capped at 1.0
    """
    return min(1.0, stable_frames / REQUIRED_STABLE_FRAMES)


def compute_intention_score(finger_score, stability_score):
    """
    INTENTION RECOGNITION EQUATION:

        intention_score = finger_score × stability_score

    Both must be high for the robot to act:
    - Low finger_score  = unclear gesture     → score stays low
    - Low stability     = fleeting gesture    → score stays low
    - Both high         = clear sustained intent → score is high → robot acts

    This models real HRI: the human must show CLEAR and SUSTAINED intent.
    """
    return round(finger_score * stability_score, 3)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PLANNING / DECISION MAKING
# Should the robot act right now?
# ─────────────────────────────────────────────────────────────────────────────

def should_trigger(intention_score, op_index, operation, trigger_key, now):
    """
    Planning: decide if conditions are right to trigger the robot.

    All conditions must be true:
    1. intention_score >= INTENTION_THRESHOLD  (clear enough intent)
    2. operation exists in saved operations    (valid target)
    3. cooldown has passed OR new gesture      (avoid repeating)
    """
    global LAST_TRIGGER_AT, LAST_TRIGGER_KEY

    cooldown_ok = (now - LAST_TRIGGER_AT) > COOLDOWN_MS
    new_gesture = trigger_key != LAST_TRIGGER_KEY

    return (
        intention_score >= INTENTION_THRESHOLD
        and op_index    is not None
        and operation   is not None
        and (new_gesture or cooldown_ok)
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — process_frame
# Called every webcam frame by the simulator
# ─────────────────────────────────────────────────────────────────────────────

def process_frame(frame):
    global LAST_TRIGGER_AT, LAST_TRIGGER_KEY, STABILITY

    now = frame.get("timestamp_ms", 0)

    # ── PERCEPTION: is there a hand? ─────────────────────────────────────────
    hand = frame.get("primary_hand")
    if not hand:
        STABILITY        = {}
        LAST_TRIGGER_KEY = None
        return {
            "label":      "No hand detected",
            "confidence": 0.0,
            "debug_text": [
                "Show one hand to the webcam.",
                "Raise 1, 2, 3, or 4 fingers to select an operation.",
                "Hold the gesture steady until the robot triggers.",
            ],
        }

    # ── PERCEPTION: read finger count ────────────────────────────────────────
    hand_label = hand.get("handedness") or hand.get("viewerSide") or "Unknown"
    count      = read_finger_count(hand)

    # ── INTENTION EQUATION: finger score ─────────────────────────────────────
    finger_score, op_index = compute_finger_score(count)

    # ── INTENTION EQUATION: stability score ──────────────────────────────────
    stability_key = f"{hand_label}:{count}"
    stable_frames = update_stability(stability_key)
    stability_score = compute_stability_score(stable_frames)

    # ── INTENTION EQUATION: final score ──────────────────────────────────────
    intention_score = compute_intention_score(finger_score, stability_score)

    # ── PLANNING: fetch operation ─────────────────────────────────────────────
    operation   = operation_by_index(frame, op_index)
    trigger_key = f"{op_index}:{hand_label}"

    # ── PLANNING: decide whether to trigger ──────────────────────────────────
    trigger = should_trigger(intention_score, op_index, operation, trigger_key, now)

    if trigger:
        LAST_TRIGGER_AT  = now
        LAST_TRIGGER_KEY = trigger_key

    # ── SECTION 4: ROBOT ACTION ───────────────────────────────────────────────
    # Handled by returning trigger_operation_id to the simulator

    # ── SECTION 5: FEEDBACK ───────────────────────────────────────────────────
    if trigger and operation:
        status = f"TRIGGERED → {operation['name']} ✅"
    elif op_index is not None and operation:
        status = f"{hand_label} hand: {count} finger(s) → {operation['name']} — hold steady..."
    elif count == 0:
        status = "Fist detected — raise 1 to 4 fingers"
    else:
        status = f"{count} finger(s) detected — waiting..."

    # Progress bar showing intention score
    bar_filled = int(intention_score * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)

    return {
        # ── Robot action ──────────────────────────────────────────────────────
        "label":      status,
        "confidence": intention_score,
        "trigger_operation_id":   operation["id"]   if trigger and operation else None,
        "trigger_operation_name": operation["name"] if trigger and operation else None,
        "cooldown_ms": COOLDOWN_MS,

        # ── Feedback / debug panel ────────────────────────────────────────────
        "debug_text": [
            f"PERCEPTION   : {hand_label} hand | {count} finger(s) detected",
            f"FINGER SCORE : {finger_score:.2f}  (how clear is the gesture?)",
            f"STABILITY    : {stability_score:.2f}  ({stable_frames}/{REQUIRED_STABLE_FRAMES} frames held)",
            f"INTENTION    : {intention_score:.2f} = {finger_score:.2f} × {stability_score:.2f}  [{bar}]",
            f"THRESHOLD    : {INTENTION_THRESHOLD}  ({'✅ PASSED' if intention_score >= INTENTION_THRESHOLD else '❌ not yet'})",
            f"ROBOT ACTION : {operation['name'] if trigger and operation else 'waiting...'}",
            "────────────────────────────────────────────────────",
            "1 finger=op1 | 2=op2 | 3=op3 | 4=op4 | hold steady to trigger",
        ],
    }
