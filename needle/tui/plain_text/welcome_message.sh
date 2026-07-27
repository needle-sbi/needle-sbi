#!/usr/bin/env bash
# Disclaimer: Generated with the help of GPT4.5

WIDTH=60
LEFT_COL_WIDTH=30
RIGHT_COL_WIDTH=24
LOGO_FILE="${NEEDLE_TUI_DIR}/plain_text/needle_name_ascii.txt"


# Colors
_NEEDLE_RED='\033[0;31m'
_NEEDLE_GREEN='\033[0;32m'
_NEEDLE_ORANGE='\033[0;33m'
_NEEDLE_BLUE='\033[0;94m'
_NEEDLE_BOLD='\033[1m'
_NEEDLE_NC='\033[0m' # No Color

center_line() {
    local text="$1"
    local len=${#text}
    local pad=$(( (WIDTH - len) / 2 ))
    printf "%*s%s\n" "$pad" "" "$text"
}

print_border() {
    printf "┌"
    printf "─%.0s" $(seq 1 $((WIDTH-2)))
    printf "┐\n"
}

print_footer() {
    printf "└"
    printf "─%.0s" $(seq 1 $((WIDTH-2)))
    printf "┘\n"
}

print_empty() {
    printf "│%*s│\n" $((WIDTH-2)) ""
}

_color_code() {
    case "$1" in
        red)    printf '%s' "$_NEEDLE_RED" ;;
        green)  printf '%s' "$_NEEDLE_GREEN" ;;
        blue)   printf '%s' "$_NEEDLE_BLUE" ;;
        orange) printf '%s' "$_NEEDLE_ORANGE" ;;
        bold)   printf '%s' "$_NEEDLE_BOLD" ;;
        *)      printf '' ;;
    esac
}

# print_text_line PLAIN_TEXT [COLOR]
# COLOR is one of: none, red, green, blue, orange, bold. Padding is always
# computed from PLAIN_TEXT so ANSI escapes never throw off alignment.
print_text_line() {
    local plain="$1"
    local color="${2:-none}"
    local pad=$(( WIDTH - 4 - ${#plain} ))
    ((pad < 0)) && pad=0

    local code
    code="$(_color_code "$color")"

    if [[ -n "$code" ]]; then
        printf "│ ${code}%s${_NEEDLE_NC}%*s │\n" "$plain" "$pad" ""
    else
        printf "│ %s%*s │\n" "$plain" "$pad" ""
    fi
}

print_header_line() {
    print_text_line "$1" "bold"
}

# print_two_col_line LEFT_TEXT LEFT_COLOR RIGHT_TEXT RIGHT_COLOR
print_two_col_line() {
    local left="$1" left_color="$2" right="$3" right_color="$4"
    ((${#left} > LEFT_COL_WIDTH)) && left="${left:0:$((LEFT_COL_WIDTH-1))}…"
    ((${#right} > RIGHT_COL_WIDTH)) && right="${right:0:$((RIGHT_COL_WIDTH-1))}…"
    local left_pad=$(( LEFT_COL_WIDTH - ${#left} ))
    ((left_pad < 0)) && left_pad=0
    local right_pad=$(( RIGHT_COL_WIDTH - ${#right} ))
    ((right_pad < 0)) && right_pad=0

    local lcode rcode
    lcode="$(_color_code "$left_color")"
    rcode="$(_color_code "$right_color")"

    printf "│ "
    if [[ -n "$lcode" ]]; then
        printf "${lcode}%s${_NEEDLE_NC}%*s" "$left" "$left_pad" ""
    else
        printf "%s%*s" "$left" "$left_pad" ""
    fi
    printf "  "
    if [[ -n "$rcode" ]]; then
        printf "${rcode}%s${_NEEDLE_NC}%*s" "$right" "$right_pad" ""
    else
        printf "%s%*s" "$right" "$right_pad" ""
    fi
    printf " │\n"
}

print_center_block() {
    while IFS= read -r line; do
        local len=${#line}
        local pad=$(( (WIDTH - 2 - len) / 2 ))
        printf "│%*s${_NEEDLE_ORANGE}%s${_NEEDLE_NC}%*s│\n" "$pad" "" "$line" "$((WIDTH-2-len-pad))" ""
    done < "$LOGO_FILE"
}

print_border
print_empty
print_center_block
print_empty

print_text_line " Welcome to NEEDLE, the workflow manager for NSBI tools"
print_empty

# Get version/environment status information from Python module
PANEL_LINES_OUTPUT=$(python3 "${NEEDLE_TUI_DIR}/components/version_info.py" --panel-lines 2>/dev/null)

if [ -n "$PANEL_LINES_OUTPUT" ]; then
    LEFT_TEXTS=()
    LEFT_COLORS=()
    RIGHT_TEXTS=()
    RIGHT_COLORS=()
    while IFS=$'\x1f' read -r side text color; do
        [ -z "$side" ] && continue
        if [ "$side" = "L" ]; then
            LEFT_TEXTS+=("$text")
            LEFT_COLORS+=("$color")
        else
            RIGHT_TEXTS+=("$text")
            RIGHT_COLORS+=("$color")
        fi
    done <<< "$PANEL_LINES_OUTPUT"

    print_header_line "Environment versions:"
    n_rows=${#LEFT_TEXTS[@]}
    ((${#RIGHT_TEXTS[@]} > n_rows)) && n_rows=${#RIGHT_TEXTS[@]}
    for ((i=0; i<n_rows; i++)); do
        print_two_col_line "  ${LEFT_TEXTS[$i]:-}" "${LEFT_COLORS[$i]:-none}" "${RIGHT_TEXTS[$i]:-}" "${RIGHT_COLORS[$i]:-none}"
    done
    print_empty
fi

print_header_line "Documentation:"
print_text_line "  needle-sbi:    https://needle-sbi.readthedocs.io" "blue"
print_text_line "  luigi:         https://luigi.readthedocs.io/" "blue"
print_text_line "  b2luigi:       https://b2luigi.belle2.org" "blue"
print_empty

print_header_line "Entry points:"
print_text_line "   law run MainTask ..."
print_text_line "   needle run ..."
print_text_line "   needle run --backend b2luigi ..."

print_empty
print_footer
