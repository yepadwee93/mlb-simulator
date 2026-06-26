"""
email_alerts.py
---------------
Sends daily top-plays email to opted-in users via Gmail SMTP.
Requires env vars: EMAIL_USER (gmail address), EMAIL_PASS (app password).

Gmail setup: Google Account → Security → 2-Step Verification → App Passwords
Generate a 16-char app password and store as EMAIL_PASS.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date

from data.db import supa

EMAIL_USER = os.getenv("EMAIL_USER", "")
EMAIL_PASS = os.getenv("EMAIL_PASS", "")


def get_alert_subscribers() -> list:
    """Return list of {username, email} for users with email_alerts=True."""
    try:
        res = supa().table("users") \
            .select("username, email") \
            .eq("email_alerts", True) \
            .not_.is_("email", "null") \
            .execute()
        return res.data or []
    except Exception:
        return []


def update_user_email(user_id: int, email: str, alerts_on: bool) -> bool:
    """Save email + alert preference for a user."""
    try:
        supa().table("users").update({
            "email": email.strip().lower(),
            "email_alerts": alerts_on,
        }).eq("id", user_id).execute()
        return True
    except Exception:
        return False


def get_user_email_settings(user_id: int) -> dict:
    """Return {email, email_alerts} for a user."""
    try:
        res = supa().table("users") \
            .select("email, email_alerts") \
            .eq("id", user_id) \
            .execute()
        if res.data:
            return res.data[0]
    except Exception:
        pass
    return {"email": None, "email_alerts": False}


def _build_html(top_plays: list, date_str: str, site_url: str) -> str:
    """Render the HTML email body."""

    def tier_color(win_pct):
        if win_pct >= 70: return "#ef5350"
        if win_pct >= 65: return "#4caf50"
        return "#ffd54f"

    plays_html = ""
    for p in top_plays:
        color = tier_color(p["win_pct"])
        ev_txt = f"+${p['ev']:.2f} EV" if p.get("ev") and p["ev"] > 0 else ""
        plays_html += f"""
        <tr>
          <td style="padding:12px 16px; border-bottom:1px solid #252836;">
            <b style="color:#e8eaf0;">{p['team']}</b>
            <span style="color:#9aa0b8; font-size:12px;"> vs {p['opponent']}</span>
          </td>
          <td style="padding:12px 16px; text-align:center; border-bottom:1px solid #252836;">
            <span style="color:{color}; font-weight:700; font-size:16px;">{p['win_pct']}%</span>
          </td>
          <td style="padding:12px 16px; text-align:center; border-bottom:1px solid #252836;">
            <span style="color:#ffd54f; font-weight:700;">{p.get('ml_odds','—')}</span>
          </td>
          <td style="padding:12px 16px; text-align:center; border-bottom:1px solid #252836;">
            <span style="color:#81c784;">{ev_txt}</span>
          </td>
        </tr>"""

    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="background:#0d1120; color:#e8eaf0; font-family:'Segoe UI',Arial,sans-serif;
             margin:0; padding:0;">
  <div style="max-width:560px; margin:32px auto; background:#1a1e2e;
              border-radius:14px; overflow:hidden; border:1px solid #252836;">

    <!-- Header -->
    <div style="background:#1e2235; padding:24px 28px; border-bottom:1px solid #353a52;">
      <div style="font-size:22px; font-weight:800; color:#ffffff;">
        ⚾ MLB Simulator — Daily Top Plays
      </div>
      <div style="font-size:13px; color:#9aa0b8; margin-top:4px;">{date_str}</div>
    </div>

    <!-- Body -->
    <div style="padding:24px 28px;">
      <p style="font-size:14px; color:#b0b8cc; margin-bottom:20px;">
        Today's best model edges — games where our 100k-simulation model
        gives a team ≥65% win probability.
      </p>

      {'<p style="color:#555870;font-size:14px;">No strong plays found for today — no games cross the 65% threshold.</p>' if not top_plays else f"""
      <table style="width:100%; border-collapse:collapse; font-size:14px;">
        <thead>
          <tr style="color:#9aa0b8; font-size:11px; text-transform:uppercase; letter-spacing:.5px;">
            <th style="padding:8px 16px; text-align:left; border-bottom:1px solid #353a52;">Team</th>
            <th style="padding:8px 16px; text-align:center; border-bottom:1px solid #353a52;">Model %</th>
            <th style="padding:8px 16px; text-align:center; border-bottom:1px solid #353a52;">ML Odds</th>
            <th style="padding:8px 16px; text-align:center; border-bottom:1px solid #353a52;">EV</th>
          </tr>
        </thead>
        <tbody>{plays_html}</tbody>
      </table>"""}

      <div style="margin-top:24px; padding:16px; background:#0d1120;
                  border-radius:10px; font-size:12px; color:#9aa0b8;">
        <b style="color:#b0b8cc;">How to use:</b> These are model edges, not guarantees.
        A 68% model win% means roughly 68 out of 100 sims say this team wins —
        but the real world adds variance. Always size bets with Kelly or flat units.
        <br><br>
        <a href="{site_url}" style="color:#64b5f6;">Open simulator →</a>
      </div>
    </div>

    <!-- Footer -->
    <div style="padding:16px 28px; background:#0d1120; font-size:11px; color:#555870;
                text-align:center; border-top:1px solid #252836;">
      MLB Simulator · For entertainment only, not financial advice ·
      <a href="{site_url}/settings" style="color:#555870;">Unsubscribe</a>
    </div>
  </div>
</body>
</html>"""


def send_daily_alerts(top_plays: list, site_url: str = "https://mlb-simulator-vert.vercel.app") -> dict:
    """
    Send the daily top-plays email to all opted-in subscribers.
    Returns {"sent": int, "failed": int, "skipped": int}.
    """
    if not EMAIL_USER or not EMAIL_PASS:
        return {"sent": 0, "failed": 0, "skipped": 0, "error": "EMAIL_USER/EMAIL_PASS not configured"}

    subscribers = get_alert_subscribers()
    if not subscribers:
        return {"sent": 0, "failed": 0, "skipped": 0}

    today_str = date.today().strftime("%A, %B %d %Y")
    html_body = _build_html(top_plays, today_str, site_url)
    subject   = f"⚾ MLB Top Plays — {today_str}"

    sent = failed = 0

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            for sub in subscribers:
                email = sub.get("email", "").strip()
                if not email or "@" not in email:
                    continue
                try:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"]    = f"MLB Simulator <{EMAIL_USER}>"
                    msg["To"]      = email
                    msg.attach(MIMEText(html_body, "html"))
                    server.sendmail(EMAIL_USER, email, msg.as_string())
                    sent += 1
                except Exception:
                    failed += 1
    except Exception as e:
        return {"sent": sent, "failed": failed, "skipped": 0, "error": str(e)}

    return {"sent": sent, "failed": failed, "skipped": 0}
