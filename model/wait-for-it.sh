#!/bin/bash
# wait-for-it.sh: Wait for a service to be available
# Usage: ./wait-for-it.sh host:port [-s] [-t timeout] [-- command args]
# -s: Strict mode - exit with non-zero if timeout is reached
# -t: Timeout in seconds (default: 15)
# -- command args: Command with args to run after the service is available

WAITFORIT_cmdname=${0##*/}

# Default timeout of 15 seconds
WAITFORIT_TIMEOUT=15
WAITFORIT_STRICT=0
WAITFORIT_CHILD=0
WAITFORIT_QUIET=0

function usage {
    cat << USAGE >&2
Usage:
    $WAITFORIT_cmdname host:port [-s] [-t timeout] [-- command args]
    -h HOST | --host=HOST       Host or IP under test
    -p PORT | --port=PORT       TCP port under test
    -s | --strict               Only execute subcommand if the test succeeds
    -q | --quiet                Don't output any status messages
    -t TIMEOUT | --timeout=TIMEOUT
                                Timeout in seconds, zero for no timeout
    -- COMMAND ARGS             Execute command with args after the test finishes
USAGE
    exit 1
}

function wait_for {
    if [[ $WAITFORIT_TIMEOUT -gt 0 ]]; then
        echo "Waiting up to $WAITFORIT_TIMEOUT seconds for $WAITFORIT_HOST:$WAITFORIT_PORT to be available"
    else
        echo "Waiting for $WAITFORIT_HOST:$WAITFORIT_PORT without a timeout"
    fi
    
    start_ts=$(date +%s)
    while :
    do
        if [[ $WAITFORIT_ISBUSY -eq 1 ]]; then
            nc -z $WAITFORIT_HOST $WAITFORIT_PORT
            result=$?
        else
            (echo > /dev/tcp/$WAITFORIT_HOST/$WAITFORIT_PORT) >/dev/null 2>&1
            result=$?
        fi
        
        if [[ $result -eq 0 ]]; then
            end_ts=$(date +%s)
            echo "$WAITFORIT_HOST:$WAITFORIT_PORT is available after $((end_ts - start_ts)) seconds"
            break
        fi
        
        sleep 1
        
        if [[ $WAITFORIT_TIMEOUT -gt 0 && $(date +%s) -gt $((start_ts + WAITFORIT_TIMEOUT)) ]]; then
            echo "Timeout reached after waiting $WAITFORIT_TIMEOUT seconds for $WAITFORIT_HOST:$WAITFORIT_PORT"
            exit 1
        fi
    done
    
    return 0
}

while [[ $# -gt 0 ]]
do
    case "$1" in
        *:* )
        hostport=(${1//:/ })
        WAITFORIT_HOST=${hostport[0]}
        WAITFORIT_PORT=${hostport[1]}
        shift 1
        ;;
        --host=*)
        WAITFORIT_HOST="${1#*=}"
        shift 1
        ;;
        --port=*)
        WAITFORIT_PORT="${1#*=}"
        shift 1
        ;;
        -q | --quiet)
        WAITFORIT_QUIET=1
        shift 1
        ;;
        -s | --strict)
        WAITFORIT_STRICT=1
        shift 1
        ;;
        -t)
        WAITFORIT_TIMEOUT="$2"
        if [[ $WAITFORIT_TIMEOUT == "" ]]; then break; fi
        shift 2
        ;;
        --timeout=*)
        WAITFORIT_TIMEOUT="${1#*=}"
        shift 1
        ;;
        --)
        shift
        WAITFORIT_CLI=("$@")
        break
        ;;
        --help)
        usage
        ;;
        *)
        echo "Unknown argument: $1"
        usage
        ;;
    esac
done

if [[ "$WAITFORIT_HOST" == "" || "$WAITFORIT_PORT" == "" ]]; then
    echo "Error: you need to provide a host and port to test."
    usage
fi

WAITFORIT_ISBUSY=0
if [[ $WAITFORIT_TIMEOUT -gt 0 ]]; then
    WAITFORIT_ISBUSY=1
fi

wait_for

WAITFORIT_RESULT=$?
if [[ $WAITFORIT_STRICT -eq 1 ]]; then
    if [[ $WAITFORIT_RESULT -ne 0 ]]; then
        exit $WAITFORIT_RESULT
    fi
fi

if [[ ${#WAITFORIT_CLI[@]} -ne 0 ]]; then
    if [[ $WAITFORIT_RESULT -ne 0 && $WAITFORIT_STRICT -eq 1 ]]; then
        exit $WAITFORIT_RESULT
    fi
    exec "${WAITFORIT_CLI[@]}"
else
    exit 0
fi 