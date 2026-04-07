def get_signal_time(count):
    if count < 5:
        return 15, "Low"
    elif count < 15:
        return 30, "Medium"
    elif count < 30:
        return 45, "High"
    else:
        return 60, "Very High"