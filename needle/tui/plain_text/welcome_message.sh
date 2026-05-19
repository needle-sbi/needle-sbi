#!/usr/bin/env bash
# Disclaimer: Generated with the help of GPT4.5

WIDTH=60
LOGO_FILE="${NEEDLE_TUI_DIR}/plain_text/needle_name_ascii.txt"


# Colors
_NEEDLE_RED='\033[0;31m'
_NEEDLE_GREEN='\033[0;32m'
_NEEDLE_ORANGE='\033[0;33m'
_NEEDLE_NC='\033[0m' # No Color
_NEEDLE_BLUE='\033[0;34m'

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

print_text_line() {
    local text="$1"
    printf "│ %-*s │\n" $((WIDTH-4)) "$text"
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

# Get version information from Python module
VERSIONS_OUTPUT=$(python3 "${NEEDLE_TUI_DIR}/components/version_info.py" --text 2>/dev/null)

if [ -n "$VERSIONS_OUTPUT" ]; then
    print_text_line "Environment Versions:"
    while IFS= read -r version_line; do
        print_text_line "  $version_line"
    done <<< "$VERSIONS_OUTPUT"
    print_empty
fi

print_text_line "Docs:        https://..."
print_text_line "Examples:    https://..."
print_text_line "Luigi:       https://luigi.readthedocs.io/en/stable/"
print_empty

print_text_line "Useful commands:"
print_text_line "  * law index"
print_text_line "  * law --help"
print_text_line "  * law run MainTask --config-file "
print_text_line "  * law run DownstreamTask --config-file "

print_empty
print_footer
