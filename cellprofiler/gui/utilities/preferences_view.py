def secs_to_timestr(duration):
    dur = int(round(duration))
    hours = dur // (60 * 60)
    rest = dur % (60 * 60)
    minutes = rest // 60
    rest %= 60
    seconds = rest
    minutes = ("%02d:" if hours > 0 else "%d:") % minutes
    hours = "%d:" % (hours,) if hours > 0 else ""
    seconds = "%02d" % seconds
    return hours + minutes + seconds
