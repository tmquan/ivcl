"""notioncode: code in the colours Notion gives it.

Notion colours code with Prism. This is a port of Prism's tokenizer and of three of its grammars
(Python, Bash, YAML), so that a page can show a code block exactly as Notion would:

    highlight(code, "bash")      HTML: the code with one <span class="token ..."> around each token

simscene.colour() uses it for the code blocks of a notebook page.
"""
import html, re

A, I, M = re.ASCII, re.IGNORECASE, re.MULTILINE


class Token:
    __slots__ = ("type", "content", "alias", "length")

    def __init__(self, type_, content, alias, matched):
        self.type, self.content, self.alias, self.length = type_, content, alias, len(matched)


class _Node:
    __slots__ = ("value", "prev", "next")

    def __init__(self, value, prev, nxt):
        self.value, self.prev, self.next = value, prev, nxt


class _List:
    def __init__(self):
        self.head = _Node(None, None, None)
        self.tail = _Node(None, self.head, None)
        self.head.next = self.tail
        self.length = 0


def _add_after(lst, node, value):
    nxt = node.next
    new = _Node(value, node, nxt)
    node.next = new
    nxt.prev = new
    lst.length += 1
    return new


def _remove_range(lst, node, count):
    nxt, i = node.next, 0
    while i < count and nxt is not lst.tail:
        nxt = nxt.next
        i += 1
    node.next = nxt
    nxt.prev = node
    lst.length -= i


def _len(value):
    return value.length if isinstance(value, Token) else len(value)


def _match(pattern, pos, text, lookbehind):
    m = pattern.search(text, pos)
    if not m:
        return None
    index, matched = m.start(), m.group(0)
    if lookbehind and m.group(1):
        index += len(m.group(1))
        matched = matched[len(m.group(1)):]
    return index, matched


def _match_grammar(text, lst, grammar, start_node, start_pos, rematch):
    for token, patterns in grammar.items():
        if not patterns:
            continue
        if not isinstance(patterns, list):
            patterns = [patterns]
        for j, obj in enumerate(patterns):
            if rematch and rematch["cause"] == f"{token},{j}":
                return
            if isinstance(obj, dict):
                pattern, inside, alias = obj["pattern"], obj.get("inside"), obj.get("alias")
                lookbehind, greedy = bool(obj.get("lookbehind")), bool(obj.get("greedy"))
            else:
                pattern, inside, alias, lookbehind, greedy = obj, None, None, False, False
            current, pos = start_node.next, start_pos
            while current is not lst.tail:
                if rematch and pos >= rematch["reach"]:
                    break
                s = current.value
                if lst.length > len(text):
                    return
                if isinstance(s, Token):
                    pos += s.length
                    current = current.next
                    continue
                remove_count = 1
                if greedy:
                    m = _match(pattern, pos, text, lookbehind)
                    if not m or m[0] >= len(text):
                        break
                    frm, to = m[0], m[0] + len(m[1])
                    p = pos + _len(current.value)
                    while frm >= p:
                        current = current.next
                        p += _len(current.value)
                    p -= _len(current.value)
                    pos = p
                    if isinstance(current.value, Token):
                        pos += current.value.length
                        current = current.next
                        continue
                    k = current
                    while k is not lst.tail and (p < to or isinstance(k.value, str)):
                        remove_count += 1
                        p += _len(k.value)
                        k = k.next
                    remove_count -= 1
                    s = text[pos:p]
                    index, matched = m[0] - pos, m[1]
                else:
                    m = _match(pattern, 0, s, lookbehind)
                    if not m:
                        pos += len(s)
                        current = current.next
                        continue
                    index, matched = m
                before, after = s[:index], s[index + len(matched):]
                reach = pos + len(s)
                if rematch and reach > rematch["reach"]:
                    rematch["reach"] = reach
                remove_from = current.prev
                if before:
                    remove_from = _add_after(lst, remove_from, before)
                    pos += len(before)
                _remove_range(lst, remove_from, remove_count)
                wrapped = Token(token, tokenize(matched, inside) if inside else matched, alias, matched)
                current = _add_after(lst, remove_from, wrapped)
                if after:
                    _add_after(lst, current, after)
                if remove_count > 1:
                    nested = {"cause": f"{token},{j}", "reach": reach}
                    _match_grammar(text, lst, grammar, current.prev, pos, nested)
                    if rematch and nested["reach"] > rematch["reach"]:
                        rematch["reach"] = nested["reach"]
                pos += _len(current.value)
                current = current.next


def tokenize(text, grammar):
    rest = grammar.get("rest")
    if rest:                                  # Prism appends the tokens of "rest" to the grammar
        grammar = {k: v for k, v in grammar.items() if k != "rest"}
        grammar.update(rest)
    lst = _List()
    _add_after(lst, lst.head, text)
    _match_grammar(text, lst, grammar, lst.head, 0, None)
    out, node = [], lst.head.next
    while node is not lst.tail:
        out.append(node.value)
        node = node.next
    return out


def stringify(tokens):
    if isinstance(tokens, str):
        return html.escape(tokens, quote=False)
    if isinstance(tokens, list):
        return "".join(stringify(t) for t in tokens)
    classes = ["token", tokens.type]
    if tokens.alias:
        classes += tokens.alias if isinstance(tokens.alias, list) else [tokens.alias]
    return f'<span class="{" ".join(classes)}">{stringify(tokens.content)}</span>'


def plain(tokens):
    if isinstance(tokens, str):
        return tokens
    if isinstance(tokens, list):
        return "".join(plain(t) for t in tokens)
    return plain(tokens.content)


# ------------------------------------------------------------------------------ Python (prism-python.js)
_interp = {
    "format-spec": dict(pattern=re.compile(r"(:)[^:(){}]+(?=\}\Z)", A), lookbehind=True),
    "conversion-option": dict(pattern=re.compile(r"![sra](?=[:}]\Z)", A), alias="punctuation"),
}
PYTHON = {
    "comment": dict(pattern=re.compile(r"(^|[^\\])#.*", A), lookbehind=True, greedy=True),
    "string-interpolation": dict(
        pattern=re.compile(r"""(?:f|fr|rf)(?:(\"\"\"|''')[\s\S]*?\1|("|')(?:\\.|(?!\2)[^\\\r\n])*\2)""", A | I), greedy=True,
        inside={"interpolation": dict(pattern=re.compile(r"((?:^|[^{])(?:\{\{)*)\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}]|\{(?!\{)(?:[^{}])+\})+\})+\}", A),
                                      lookbehind=True, inside=_interp),
                "string": re.compile(r"[\s\S]+")}),
    "triple-quoted-string": dict(pattern=re.compile(r"""(?:[rub]|br|rb)?(\"\"\"|''')[\s\S]*?\1""", A | I), greedy=True, alias="string"),
    "string": dict(pattern=re.compile(r"""(?:[rub]|br|rb)?("|')(?:\\.|(?!\1)[^\\\r\n])*\1""", A | I), greedy=True),
    "function": dict(pattern=re.compile(r"((?:^|\s)def[ \t]+)[a-zA-Z_]\w*(?=\s*\()", A), lookbehind=True),
    "class-name": dict(pattern=re.compile(r"(\bclass\s+)\w+", A | I), lookbehind=True),
    "decorator": dict(pattern=re.compile(r"(^[\t ]*)@\w+(?:\.\w+)*", A | M), lookbehind=True, alias=["annotation", "punctuation"],
                      inside={"punctuation": re.compile(r"\.")}),
    "keyword": re.compile(r"\b(?:_(?=\s*:)|and|as|assert|async|await|break|case|class|continue|def|del|elif|else|except|exec|finally|for|from|global|if|import|in|is|lambda|match|nonlocal|not|or|pass|print|raise|return|try|while|with|yield)\b", A),
    "builtin": re.compile(r"\b(?:__import__|abs|all|any|apply|ascii|basestring|bin|bool|buffer|bytearray|bytes|callable|chr|classmethod|cmp|coerce|compile|complex|delattr|dict|dir|divmod|enumerate|eval|execfile|file|filter|float|format|frozenset|getattr|globals|hasattr|hash|help|hex|id|input|int|intern|isinstance|issubclass|iter|len|list|locals|long|map|max|memoryview|min|next|object|oct|open|ord|pow|property|range|raw_input|reduce|reload|repr|reversed|round|set|setattr|slice|sorted|staticmethod|str|sum|super|tuple|type|unichr|unicode|vars|xrange|zip)\b", A),
    "boolean": re.compile(r"\b(?:False|None|True)\b", A),
    "number": re.compile(r"\b0(?:b(?:_?[01])+|o(?:_?[0-7])+|x(?:_?[a-f0-9])+)\b|(?:\b\d+(?:_\d+)*(?:\.(?:\d+(?:_\d+)*)?)?|\B\.\d+(?:_\d+)*)(?:e[+-]?\d+(?:_\d+)*)?j?(?!\w)", A | I),
    "operator": re.compile(r"[-+%=]=?|!=|:=|\*\*?=?|//?=?|<[<=>]?|>[=>]?|[&|^~]"),
    "punctuation": re.compile(r"[{}\[\];(),.:]"),
}
_interp["rest"] = PYTHON

# ------------------------------------------------------------------------------ Bash (prism-bash.js)
_ENV = (r"\b(?:BASH|BASHOPTS|BASH_ALIASES|BASH_ARGC|BASH_ARGV|BASH_CMDS|BASH_COMPLETION_COMPAT_DIR|BASH_LINENO|BASH_REMATCH|BASH_SOURCE|BASH_VERSINFO|BASH_VERSION|"
        r"COLORTERM|COLUMNS|COMP_WORDBREAKS|DBUS_SESSION_BUS_ADDRESS|DEFAULTS_PATH|DESKTOP_SESSION|DIRSTACK|DISPLAY|EUID|GDMSESSION|GDM_LANG|GNOME_KEYRING_CONTROL|"
        r"GNOME_KEYRING_PID|GPG_AGENT_INFO|GROUPS|HISTCONTROL|HISTFILE|HISTFILESIZE|HISTSIZE|HOME|HOSTNAME|HOSTTYPE|IFS|INSTANCE|JOB|LANG|LANGUAGE|LC_ADDRESS|LC_ALL|"
        r"LC_IDENTIFICATION|LC_MEASUREMENT|LC_MONETARY|LC_NAME|LC_NUMERIC|LC_PAPER|LC_TELEPHONE|LC_TIME|LESSCLOSE|LESSOPEN|LINES|LOGNAME|LS_COLORS|MACHTYPE|MAILCHECK|"
        r"MANDATORY_PATH|NO_AT_BRIDGE|OLDPWD|OPTERR|OPTIND|ORBIT_SOCKETDIR|OSTYPE|PAPERSIZE|PATH|PIPESTATUS|PPID|PS1|PS2|PS3|PS4|PWD|RANDOM|REPLY|SECONDS|SELINUX_INIT|"
        r"SESSION|SESSIONTYPE|SESSION_MANAGER|SHELL|SHELLOPTS|SHLVL|SSH_AUTH_SOCK|TERM|UID|UPSTART_EVENTS|UPSTART_INSTANCE|UPSTART_JOB|UPSTART_SESSION|USER|WINDOWID|"
        r"XAUTHORITY|XDG_CONFIG_DIRS|XDG_CURRENT_DESKTOP|XDG_DATA_DIRS|XDG_GREETER_DATA_DIR|XDG_MENU_PREFIX|XDG_RUNTIME_DIR|XDG_SEAT|XDG_SEAT_PATH|XDG_SESSION_DESKTOP|"
        r"XDG_SESSION_ID|XDG_SESSION_PATH|XDG_SESSION_TYPE|XDG_VTNR|XMODIFIERS)\b")
_START = r"(^|[\s;|&]|[<>]\()"
_END = r"(?=\Z|[)\s;|&])"
_after_heredoc = dict(pattern=re.compile(r"""(^(["']?)\w+\2)[ \t]+\S.*""", A), lookbehind=True, alias="punctuation", inside=None)
_command_substitution_inside = {"variable": re.compile(r"^\$\(|^`|\)\Z|`\Z")}
_in_string = {
    "bash": _after_heredoc,
    "environment": dict(pattern=re.compile(r"\$" + _ENV, A), alias="constant"),
    "variable": [
        dict(pattern=re.compile(r"\$?\(\([\s\S]+?\)\)", A), greedy=True, inside={
            "variable": [dict(pattern=re.compile(r"(^\$\(\([\s\S]+)\)\)", A), lookbehind=True), re.compile(r"^\$\(\(")],
            "number": re.compile(r"\b0x[\dA-Fa-f]+\b|(?:\b\d+(?:\.\d*)?|\B\.\d+)(?:[Ee]-?\d+)?", A),
            "operator": re.compile(r"--|\+\+|\*\*=?|<<=?|>>=?|&&|\|\||[=!+\-*/%<>^&|]=?|[?~:]"),
            "punctuation": re.compile(r"\(\(?|\)\)?|,|;")}),
        dict(pattern=re.compile(r"\$\((?:\([^)]+\)|[^()])+\)|`[^`]+`", A), greedy=True, inside=_command_substitution_inside),
        dict(pattern=re.compile(r"\$\{[^}]+\}", A), greedy=True, inside={
            "operator": re.compile(r":[-=?+]?|[!/]|##?|%%?|\^\^?|,,?"),
            "punctuation": re.compile(r"[\[\]]"),
            "environment": dict(pattern=re.compile(r"(\{)" + _ENV, A), lookbehind=True, alias="constant")}),
        re.compile(r"\$(?:\w+|[#?*!@$])", A),
    ],
    "entity": re.compile(r"""\\(?:[abceEfnrtv\\"]|O?[0-7]{1,3}|U[0-9a-fA-F]{8}|u[0-9a-fA-F]{4}|x[0-9a-fA-F]{1,2})""", A),
}
BASH = {
    "shebang": dict(pattern=re.compile(r"^#!\s*/.*", A), alias="important"),
    "comment": dict(pattern=re.compile(r"""(^|[^"{\\$])#.*""", A), lookbehind=True),
    "function-name": [
        dict(pattern=re.compile(r"(\bfunction\s+)[\w-]+(?=(?:\s*\(?:\s*\))?\s*\{)", A), lookbehind=True, alias="function"),
        dict(pattern=re.compile(r"\b[\w-]+(?=\s*\(\s*\)\s*\{)", A), alias="function"),
    ],
    "for-or-select": dict(pattern=re.compile(r"(\b(?:for|select)\s+)\w+(?=\s+in\s)", A), alias="variable", lookbehind=True),
    "assign-left": dict(pattern=re.compile(_START + r"\w+(?:\.\w+)*(?=\+?=)", A), lookbehind=True, alias="variable",
                        inside={"environment": dict(pattern=re.compile(_START + _ENV, A), lookbehind=True, alias="constant")}),
    "parameter": dict(pattern=re.compile(r"(^|\s)-{1,2}(?:\w+:[+-]?)?\w+(?:\.\w+)*(?=[=\s]|\Z)", A), alias="variable", lookbehind=True),
    "string": [
        dict(pattern=re.compile(r"((?:^|[^<])<<-?\s*)(\w+)\s[\s\S]*?(?:\r?\n|\r)\2", A), lookbehind=True, greedy=True, inside=_in_string),
        dict(pattern=re.compile(r"""((?:^|[^<])<<-?\s*)(["'])(\w+)\2\s[\s\S]*?(?:\r?\n|\r)\3""", A), lookbehind=True, greedy=True, inside={"bash": _after_heredoc}),
        dict(pattern=re.compile(r"""(^|[^\\](?:\\\\)*)"(?:\\[\s\S]|\$\([^)]+\)|\$(?!\()|`[^`]+`|[^"\\`$])*\"""", A), lookbehind=True, greedy=True, inside=_in_string),
        dict(pattern=re.compile(r"(^|[^$\\])'[^']*'", A), lookbehind=True, greedy=True),
        dict(pattern=re.compile(r"\$'(?:[^'\\]|\\[\s\S])*'", A), greedy=True, inside={"entity": _in_string["entity"]}),
    ],
    "environment": dict(pattern=re.compile(r"\$?" + _ENV, A), alias="constant"),
    "variable": _in_string["variable"],
    "function": dict(pattern=re.compile(_START + r"(?:add|apropos|apt|apt-cache|apt-get|aptitude|aspell|automysqlbackup|awk|basename|bash|bc|bconsole|bg|bzip2|cal|cargo|cat|cfdisk|chgrp|chkconfig|chmod|chown|chroot|cksum|clear|cmp|column|comm|composer|cp|cron|crontab|csplit|curl|cut|date|dc|dd|ddrescue|debootstrap|df|diff|diff3|dig|dir|dircolors|dirname|dirs|dmesg|docker|docker-compose|du|egrep|eject|env|ethtool|expand|expect|expr|fdformat|fdisk|fg|fgrep|file|find|fmt|fold|format|free|fsck|ftp|fuser|gawk|git|gparted|grep|groupadd|groupdel|groupmod|groups|grub-mkconfig|gzip|halt|head|hg|history|host|hostname|htop|iconv|id|ifconfig|ifdown|ifup|import|install|ip|java|jobs|join|kill|killall|less|link|ln|locate|logname|logrotate|look|lpc|lpr|lprint|lprintd|lprintq|lprm|ls|lsof|lynx|make|man|mc|mdadm|mkconfig|mkdir|mke2fs|mkfifo|mkfs|mkisofs|mknod|mkswap|mmv|more|most|mount|mtools|mtr|mutt|mv|nano|nc|netstat|nice|nl|node|nohup|notify-send|npm|nslookup|op|open|parted|passwd|paste|pathchk|ping|pkill|pnpm|podman|podman-compose|popd|pr|printcap|printenv|ps|pushd|pv|quota|quotacheck|quotactl|ram|rar|rcp|reboot|remsync|rename|renice|rev|rm|rmdir|rpm|rsync|scp|screen|sdiff|sed|sendmail|seq|service|sftp|sh|shellcheck|shuf|shutdown|sleep|slocate|sort|split|ssh|stat|strace|su|sudo|sum|suspend|swapon|sync|sysctl|tac|tail|tar|tee|time|timeout|top|touch|tr|traceroute|tsort|tty|umount|uname|unexpand|uniq|units|unrar|unshar|unzip|update-grub|uptime|useradd|userdel|usermod|users|uudecode|uuencode|v|vcpkg|vdir|vi|vim|virsh|vmstat|wait|watch|wc|wget|whereis|which|who|whoami|write|xargs|xdg-open|yarn|yes|zenity|zip|zsh|zypper)" + _END, A), lookbehind=True),
    "keyword": dict(pattern=re.compile(_START + r"(?:case|do|done|elif|else|esac|fi|for|function|if|in|select|then|until|while)" + _END, A), lookbehind=True),
    "builtin": dict(pattern=re.compile(_START + r"(?:\.|:|alias|bind|break|builtin|caller|cd|command|continue|declare|echo|enable|eval|exec|exit|export|getopts|hash|help|let|local|logout|mapfile|printf|pwd|read|readarray|readonly|return|set|shift|shopt|source|test|times|trap|type|typeset|ulimit|umask|unalias|unset)" + _END, A),
                    lookbehind=True, alias="class-name"),
    "boolean": dict(pattern=re.compile(_START + r"(?:false|true)" + _END, A), lookbehind=True),
    "file-descriptor": dict(pattern=re.compile(r"\B&\d\b", A), alias="important"),
    "operator": dict(pattern=re.compile(r"\d?<>|>\||\+=|=[=~]?|!=?|<<[<-]?|[&\d]?>>|\d[<>]&?|[<>][&=]?|&[>&]?|\|[&|]?", A),
                     inside={"file-descriptor": dict(pattern=re.compile(r"^\d", A), alias="important")}),
    "punctuation": re.compile(r"\$?\(\(?|\)\)?|\.\.|[{}\[\];\\]"),
    "number": dict(pattern=re.compile(r"(^|\s)(?:[1-9]\d*|0)(?:[.,]\d+)?\b", A), lookbehind=True),
}
_after_heredoc["inside"] = BASH
for _name in ("comment", "function-name", "for-or-select", "assign-left", "parameter", "string", "environment", "function", "keyword", "builtin",
              "boolean", "file-descriptor", "operator", "punctuation", "number"):
    _command_substitution_inside[_name] = BASH[_name]

# ------------------------------------------------------------------------------ YAML (prism-yaml.js)
_ANCHOR = r"[*&][^\s\[\]{},]+"
_TAG = r"!(?:<[\w\-%#;/?:@&=+$,.!~*'()\[\]]+>|(?:[a-zA-Z\d-]*!)?[\w\-%#;/?:@&=+$.~*'()]+)?"
_PROPS = "(?:" + _TAG + r"(?:[ \t]+" + _ANCHOR + ")?|" + _ANCHOR + r"(?:[ \t]+" + _TAG + ")?)"
_PLAIN = r"[^\s\x00-\x08\x0e-\x1f,\[\]{}\x7f-\x84\x86-\x9f\ufffe\uffff]"
_PLAIN_KEY = (r"(?:[^\s\x00-\x08\x0e-\x1f!\"#%&'*,\-:>?@\[\]`{|}\x7f-\x84\x86-\x9f\ufffe\uffff]|[?:-]" + _PLAIN + r")(?:[ \t]*(?:(?![#:])" + _PLAIN + "|:" + _PLAIN + "))*")
_YSTRING = r"""\"(?:[^"\\\r\n]|\\.)*\"|'(?:[^'\\\r\n]|\\.)*'"""


def _value(value, flags=0):
    return re.compile(r"([:\-,\[{]\s*(?:\s" + _PROPS + r"[ \t]+)?)(?:" + value + r")(?=[ \t]*(?:$|,|\]|\}|(?:[\r\n]\s*)?#))", flags | M | A)


YAML = {
    "scalar": dict(pattern=re.compile(r"([\-:]\s*(?:\s" + _PROPS + r"[ \t]+)?[|>])[ \t]*(?:((?:\r?\n|\r)[ \t]+)\S[^\r\n]*(?:\2[^\r\n]+)*)", A), lookbehind=True, alias="string"),
    "comment": re.compile(r"#.*"),
    "key": dict(pattern=re.compile(r"((?:^|[:\-,\[{\r\n?])[ \t]*(?:" + _PROPS + r"[ \t]+)?)(?:" + _PLAIN_KEY + "|" + _YSTRING + r")(?=\s*:\s)", A),
                lookbehind=True, greedy=True, alias="atrule"),
    "directive": dict(pattern=re.compile(r"(^[ \t]*)%.+", M), lookbehind=True, alias="important"),
    "datetime": dict(pattern=_value(r"\d{4}-\d\d?-\d\d?(?:[tT]|[ \t]+)\d\d?:\d{2}:\d{2}(?:\.\d*)?(?:[ \t]*(?:Z|[-+]\d\d?(?::\d{2})?))?|\d{4}-\d{2}-\d{2}|\d\d?:\d{2}(?::\d{2}(?:\.\d*)?)?"),
                     lookbehind=True, alias="number"),
    "boolean": dict(pattern=_value(r"false|true", I), lookbehind=True, alias="important"),
    "null": dict(pattern=_value(r"null|~", I), lookbehind=True, alias="important"),
    "string": dict(pattern=_value(_YSTRING), lookbehind=True, greedy=True),
    "number": dict(pattern=_value(r"[+-]?(?:0x[\da-f]+|0o[0-7]+|(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|\.inf|\.nan)", I), lookbehind=True),
    "tag": re.compile(_TAG, A),
    "important": re.compile(_ANCHOR, A),
    "punctuation": re.compile(r"---|[:\[\]{}\-,|>?]|\.\.\."),
}

# Notion's names for the languages, as it shows them on a code block and writes them in Markdown
LANGUAGES = {"python": ("Python", PYTHON), "bash": ("Bash", BASH), "yaml": ("YAML", YAML), "plain text": ("Plain Text", None)}


def highlight(code, language):
    """HTML for a code block: the code with Prism's token spans. Raises KeyError for an unknown language."""
    label, grammar = LANGUAGES[language]
    if grammar is None:
        return html.escape(code, quote=False)
    tokens = tokenize(code, grammar)
    assert plain(tokens) == code, "tokenizing must not change the code"
    return stringify(tokens)
