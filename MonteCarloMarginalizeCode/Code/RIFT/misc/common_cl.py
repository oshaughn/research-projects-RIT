def parse_cl_key_value(params):
    """
    Convenience in parsing out parameter arrays in the form of something=something_else:
        --channel-name "H1=FAKE-STRAIN"
    """
    return dict([val.split("=") for val in params])
