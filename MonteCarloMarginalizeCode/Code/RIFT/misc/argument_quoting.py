"""Shell-to-HTCondor argument quoting shared by RIFT DAG backends."""
import shlex


def _double_up_quotes(instr):
    return instr.replace("'", "''").replace('"', '""')
def quote_arguments(args):
    """Quote a string or list of strings using the Condor submit file "new" argument quoting rules.

    Returns
    -------
    str
        The arguments in a quoted form.

    Warnings
    --------
    You will need to surround the result in double-quotes before using it in
    the Arguments attribute.

    Examples
    --------
    >>> quote_arguments(["3", "simple", "arguments"])
    '3 simple arguments'
    >>> quote_arguments(["one", "two with spaces", "three"])
    'one \'two with spaces\' three'
    >>> quote_arguments(["one", "\"two\"", "spacy 'quoted' argument"])
    'one ""two"" \'spacey \'\'quoted\'\' argument\''
    """
    if isinstance(args, str):
        args_list = [args]
    else:
        args_list = args

    quoted_args = []
    for a in args_list:
        if "\n" in a or "\r" in a:
            raise ValueError("Condor arguments cannot contain literal line breaks")
        qa = _double_up_quotes(a)
        if not qa or any(char.isspace() for char in qa) or "'" in qa:
            qa = "'" + qa + "'"
        quoted_args.append(qa)
    return " ".join(quoted_args)

def safely_quote_arg_str(arg_str):
    """Translate shell-style argument text to Condor's new argument syntax.

    Shell quoting describes the argument boundaries; it is not part of argv.
    Parse those boundaries before applying Condor escaping, including multiple
    dictionaries, nested string values and explicitly empty arguments.
    """
    return quote_arguments(shlex.split(arg_str))


