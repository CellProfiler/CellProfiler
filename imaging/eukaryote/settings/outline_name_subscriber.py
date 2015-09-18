class OutlineNameSubscriber(ImageNameSubscriber):
    '''A setting that subscribes to the list of available object outline names
    '''

    def __init__(self, text, value="None", can_be_blank=False,
                 blank_text=LEAVE_BLANK, *args, **kwargs):
        super(OutlineNameSubscriber, self).__init__(text,
                                                    value, can_be_blank,
                                                    blank_text, *args,
                                                    **kwargs)

    def matches(self, setting):
        '''Only match OutlineNameProvider variables'''
        return isinstance(setting, OutlineNameProvider)
