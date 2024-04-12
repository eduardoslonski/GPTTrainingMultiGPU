def format_magnitude(num):
    magnitude = 0
    while num >= 1_000:
        magnitude += 1
        num /= 1_000
    return f'{round(num, 2)}{["", "K", "M", "B", "T"][magnitude]}'

def format_lr(num):
    if num == int(num):
        return str(int(num))
    else:
        formatted = '{:.1e}'.format(num)
        without_zeros = formatted.rstrip('0').rstrip('.')
        base, exp = without_zeros.split('e')
        exp = str(int(exp))
        return f"{base}e{exp}"