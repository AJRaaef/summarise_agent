def format_number(x):
    """
    Format number with commas and two decimals
    """
    try:
        return f"{x:,.2f}"
    except:
        return str(x)
