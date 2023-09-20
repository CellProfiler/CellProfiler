def filter_duplicate_names(name_list):
    """remove any repeated names from a list of (name, ...) keeping the last occurrence."""
    name_dict = dict(list(zip((n[0] for n in name_list), name_list)))

    return [name_dict[n[0]] for n in name_list]


def get_name_providers(pipeline, last_setting):
    """Scan the pipeline to find name providers matching the name given in the setting
    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    returns a list of providers that provide a correct "thing" with the
    same name as that of the subscriber
    """
    from ..setting.text.alphanumeric.name import Name

    choices = []

    for module in pipeline.modules(False):
        module_choices = []

        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return choices

            if (
                isinstance(setting, Name)
                and setting != "Do not use"
                and module.enabled
                and last_setting.matches(setting)
                and setting.value == last_setting.value
            ):
                module_choices.append(setting)

        choices += module_choices

    assert False, "Setting not among visible settings in pipeline"


def get_name_provider_choices(pipeline, last_setting, group):
    """Scan the pipeline to find name providers for the given group

    pipeline - pipeline to scan
    last_setting - scan the modules in order until you arrive at this setting
    group - the name of the group of providers to scan
    returns a list of tuples, each with (provider name, module name, module number)
    """
    from ..setting.text.alphanumeric.name import Name

    choices = []

    for module in pipeline.modules(False):
        module_choices = [
            (
                other_name,
                module.module_name,
                module.module_num,
                module.is_input_module(),
            )
            for other_name in module.other_providers(group)
        ]

        for setting in module.visible_settings():
            if setting.key() == last_setting.key():
                return filter_duplicate_names(choices)

            if (
                isinstance(setting, Name)
                and module.enabled
                and setting != "Do not use"
                and last_setting.matches(setting)
            ):
                module_choices.append(
                    (
                        setting.value,
                        module.module_name,
                        module.module_num,
                        module.is_input_module(),
                    )
                )

        choices += module_choices

    assert False, "Setting not among visible settings in pipeline"
