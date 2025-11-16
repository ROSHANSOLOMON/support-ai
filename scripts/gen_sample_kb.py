# scripts/gen_sample_kb.py
import os

os.makedirs("data/kb", exist_ok=True)

kb = {
    'reset_password.txt': 'To reset your password, go to Settings -> Account -> Reset Password. A reset link is sent to your registered email.',
    'login_error.txt': 'If login returns error 401, verify username/password. If using SSO, ensure your token is valid.',
    'payment_failed.txt': 'Payment failed errors occur due to expired card or insufficient funds.',
    'app_crash_start.txt': 'If the app crashes on start, try clearing cache and reinstalling.',
    'two_factor.txt': 'Two-factor sends a 6-digit code. If not received, request a new one.',
    'email_notifications.txt': 'Manage email notifications from Settings -> Notifications.',
    'subscription_cancel.txt': 'Cancel subscription in Billing -> Cancel Subscription.',
    'data_export.txt': 'Export data from Settings -> Data & Privacy -> Export Data.',
    'api_rate_limit.txt': 'API rate limit is 600 req/min. Use exponential backoff.',
    'integration_docs.txt': 'API integration uses Authorization: Bearer <API_KEY>.',
    'install_windows.txt': 'Download installer.exe and allow in Windows Defender if blocked.',
    'install_mac.txt': 'Download .dmg, drag to Applications, then open.',
    'error_503.txt': '503 happens during maintenance. Retry after a few minutes.',
    'privacy_policy.txt': 'We collect minimal data required for functionality.',
    'backup_restore.txt': 'Restore backup via Settings -> Backup -> Restore.',
    'feature_flag.txt': 'Feature flags roll out gradually across accounts.',
    'password_policy.txt': 'Passwords require 8 chars with letters and numbers.',
    'browser_support.txt': 'Supported browsers: Chrome, Firefox, Safari, Edge.',
    'analytics_reporting.txt': 'Enable analytics for anonymized usage metrics.',
    'contact_support.txt': 'Contact support@example.com or open ticket in-app.'
}

for fn, text in kb.items():
    with open(os.path.join("data/kb", fn), "w", encoding="utf-8") as f:
        f.write(text)

print("Sample KB created with", len(kb), "files in data/kb/")
