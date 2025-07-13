def col_mapping_hydra_to_pf(mapping_dict: dict) -> dict:
    output = {}
    for k, v in mapping_dict.items():
        # need to escape the curly braces
        # prompt flow expects a mapping like:
        #   record_id: ${data.id}
        output[k] = f"${{{v}}}"

    return output


def col_mapping_hydra_to_local(mapping_dict: dict) -> dict:
    # do nothing right now.
    # TODO: if we go back to using promptflow, fix this
    return mapping_dict

    # # For local dataframe mapping
    # output = {}
    # for k, v in mapping_dict.items():
    #     # Usually we need to discard the early part of the mapping before the final '.'
    #     output[k] = v.split('.')[-1]

    # return output



