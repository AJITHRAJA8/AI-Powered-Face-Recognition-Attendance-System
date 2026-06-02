# ============================================================
#   Smart Outing System — app.py (HARD-CODED FIREBASE VERSION)
#   • No environment variables required
#   • Reads Firebase JSON directly from serviceAccount.json
#   • Fully fixed entry/exit logic
#   • Staff dashboard "Pending" issue fixed
#   • Works perfectly with updated recognize.py
# ============================================================

import os
import json
import threading
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

# ===============================
#   INITIALIZATION (HARDCODED)
# ===============================

app = Flask(__name__)

# Load Firebase JSON directly
cred = credentials.Certificate("serviceAccount.json")

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": "https://admin-faecb-default-rtdb.firebaseio.com"  # <-- your DB URL
    },
)

root = db.reference("/")




# ===============================
#   HELPERS
# ===============================

def now_iso():
    return datetime.utcnow().isoformat() + "Z"

def parse_dt(date_str, time_str):
    try:
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except:
        return None

def get_last_event(uid, ts):
    date_key = ts.strftime("%Y-%m-%d")
    evs = root.child("face_events").child(uid).child(date_key).get() or {}
    if not evs:
        return None
    last_key = list(evs.keys())[-1]
    return evs[last_key].get("type")

# ===============================
#   LEGACY ENDPOINTS
# ===============================

@app.route("/api/attendance", methods=["POST"])
def api_attendance():
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing json"}), 400
    root.child("attendance").push(data)
    return jsonify({"ok": True})

@app.route("/api/alerts", methods=["POST"])
def api_alerts():
    data = request.get_json()
    if not data:
        return jsonify({"error": "missing json"}), 400
    root.child("alerts").push(data)
    return jsonify({"ok": True})


# =======================================================
#   FACE EVENT LOGIC (MAIN)
# =======================================================

@app.route("/api/log-face-event", methods=["POST"])
def log_face_event():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON"}), 400

    uid = data.get("uid") or "unknown"
    name = data.get("name", uid)
    confidence = float(data.get("confidence", 0))
    liveness_ok = float(data.get("liveness", 1.0)) > 0.30
    camera_id = data.get("camera_id", "CAM_1")

    raw_mode = (
        data.get("mode") or
        data.get("event_type") or
        "entry"
    ).lower()

    raw_type = (data.get("raw_type") or "face_detected").lower()
    ts_str = data.get("timestamp", now_iso())

    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", ""))
    except:
        ts = datetime.utcnow()
        ts_str = now_iso()

    # ==========================
    #   FIX: STATE-BASED LOGIC
    # ==========================

    last_event = get_last_event(uid, ts)

    if last_event == "Exit Out":
        mode = "entry"
    elif last_event == "Entry In":
        mode = "exit"
    else:
        mode = raw_mode

    event_type = "Unknown"
    lateness = "unauthorized"

    # Unauthorized / spoof
    if not liveness_ok or "unauthor" in raw_type:
        event_type = "Unauthorized"
        lateness = "unauthorized"

    else:
        # ACTIVE PERMISSION
        perms = root.child("permissions").get() or {}
        active_perm = None
        active_pid = None
        active_from = None
        active_to = None

        for pid, p in perms.items():
            if p.get("uid") != uid:
                continue
            if p.get("status") != "approved":
                continue
            if p.get("returnAt"):
                continue

            from_dt = parse_dt(p.get("fromDate", ""), p.get("fromTime", "00:00"))
            to_dt = parse_dt(p.get("toDate", ""), p.get("toTime", "23:59"))

            if active_perm is None or (to_dt and to_dt > active_to):
                active_perm = p
                active_pid = pid
                active_from = from_dt
                active_to = to_dt

        # ==========================
        #   ENTRY / EXIT FIXED
        # ==========================

        if active_perm is None:
            # No permission found
            if mode == "exit":
                event_type = "Unauthorized"
            else:
                event_type = "Entry In"
            lateness = "unauthorized"

        else:
            perm_out_at = active_perm.get("outAt")
            perm_return_at = active_perm.get("returnAt")

            # EXIT LOGIC
            if mode == "exit":
                if perm_return_at:
                    event_type = "Unauthorized"
                    lateness = "unauthorized"
                else:
                    if (active_from and ts < active_from) or (active_to and ts > active_to):
                        event_type = "Unauthorized"
                        lateness = "unauthorized"
                    else:
                        event_type = "Exit Out"
                        lateness = "on-time"

                        updates = {"status": "approved"}
                        if not perm_out_at:
                            updates["outAt"] = ts_str

                        root.child("permissions").child(active_pid).update(updates)

            # ENTRY LOGIC
            else:
                event_type = "Entry In"
                if active_to and ts > active_to:
                    lateness = "late"
                else:
                    lateness = "on-time"

                root.child("permissions").child(active_pid).update(
                    {
                        "returnAt": ts_str,
                        "returnStatus": lateness,
                        "status": "approved",
                    }
                )

    # ===============================
    #   SAVE FACE EVENT
    # ===============================

    date_key = ts.strftime("%Y-%m-%d")
    root.child("face_events").child(uid).child(date_key).push(
        {
            "name": name,
            "time": ts_str,
            "type": event_type,
            "lateness": lateness,
            "confidence": confidence,
            "camera_id": camera_id,
        }
    )

    # Analytics
    update_confidence(uid, confidence)
    update_heatmap(ts, event_type)
    behavior_check(uid)

    return jsonify({"ok": True})


# ===============================
#   ANALYTICS
# ===============================

def update_confidence(uid, conf):
    ref = root.child("analytics/confidence").child(uid)
    snap = ref.get() or {}
    old_avg = float(snap.get("avg", 0))
    count = int(snap.get("count", 0)) + 1
    avg = (old_avg * (count - 1) + conf) / count
    ref.update({"avg": avg, "count": count, "updatedAt": now_iso()})

def update_heatmap(ts, etype):
    hr = ts.strftime("%H")
    date = ts.strftime("%Y-%m-%d")
    ref = root.child("analytics/heatmap").child(date).child(hr)
    snap = ref.get() or {"in": 0, "out": 0, "unauthorized": 0}

    e = etype.lower()
    if "entry" in e:
        snap["in"] += 1
    elif "exit" in e:
        snap["out"] += 1
    else:
        snap["unauthorized"] += 1

    ref.set(snap)


# ===============================
#   BEHAVIOR CHECK
# ===============================

def behavior_check(uid):
    events = root.child("face_events").child(uid).get() or {}
    week_ago = datetime.utcnow() - timedelta(days=7)
    outs = late = unauth = 0

    for date, logs in events.items():
        for ev in logs.values():
            try:
                ts = datetime.fromisoformat(ev["time"].replace("Z", ""))
            except:
                continue

            if ts < week_ago:
                continue

            t = (ev.get("type") or "").lower()
            l = (ev.get("lateness") or "").lower()

            if "exit" in t:
                outs += 1
            if "late" in l:
                late += 1
            if "unauth" in l:
                unauth += 1

    root.child("analytics/behavior").child(uid).set(
        {
            "outs_last7": outs,
            "late_last7": late,
            "unauth_last7": unauth,
            "checkedAt": now_iso(),
        }
    )


# ===============================
#   SCHEDULER
# ===============================

def scheduler_loop():
    while True:
        try:
            scan_expiry()
        except Exception as e:
            print("Scheduler Error:", e)
        time.sleep(60)

def scan_expiry():
    perms = root.child("permissions").get() or {}
    now = datetime.utcnow()

    for pid, p in perms.items():
        if p.get("status") != "approved":
            continue
        if p.get("returnAt"):
            continue

        uid = p.get("uid")
        name = p.get("name", uid)

        from_dt = parse_dt(p.get("fromDate", ""), p.get("fromTime", "00:00"))
        to_dt = parse_dt(p.get("toDate", ""), p.get("toTime", "23:59"))

        if not to_dt:
            continue

        diff = (to_dt - now).total_seconds()

        if 0 < diff <= 900:
            root.child("alerts").push(
                {
                    "message": f"Permission expiring soon for {name}",
                    "type": "expiry_soon",
                    "createdAt": now_iso(),
                }
            )

        if diff < 0:
            returned = False
            evs = root.child("face_events").child(uid).get() or {}
            for date, logs in evs.items():
                for ev in logs.values():
                    try:
                        ts = datetime.fromisoformat(ev["time"].replace("Z", ""))
                    except:
                        continue
                    if "entry" in (ev.get("type") or "").lower() and ts >= to_dt:
                        returned = True

            if not returned:
                root.child("alerts").push(
                    {
                        "message": f"{name} has NOT returned!",
                        "type": "not_returned",
                        "createdAt": now_iso(),
                    }
                )


threading.Thread(target=scheduler_loop, daemon=True).start()


# ===============================
#   HEALTH CHECK
# ===============================

@app.route("/health")
def health():
    return "OK", 200


# ===============================
#   RUN SERVER
# ===============================

if __name__ == "__main__":
    app.run(port=5000, debug=True)
