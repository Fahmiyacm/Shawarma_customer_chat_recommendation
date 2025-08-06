import re
def is_valid_uae_number(phone):
    """Validate UAE mobile number (e.g., +971501234567 or 0501234567)."""
    pattern = r"^(\+971|0)(5[0|1|2|4|5|6|7|8|9])[0-9]{7}$"
    return bool(re.match(pattern, phone.strip()))

def format_uae_number(phone):
    """Format phone number to +971 format."""
    phone = phone.replace(" ", "")
    if phone.startswith("05"):
        phone = "+971" + phone[1:]
    return phone


def extract_session_id(session_string):
    """Extract session ID from Dialogflow session string."""
    if not session_string:
        return None
    parts = session_string.split("/")
    return parts[-1] if parts else None

def get_str_from_food_dict(food_dict):
    """Convert food dictionary to string for order summary."""
    return ", ".join([f"{qty} {item}" for item, qty in food_dict.items()])