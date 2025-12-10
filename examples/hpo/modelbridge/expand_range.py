import sys

def main():
    if len(sys.argv) < 3:
        # No args, output empty
        return

    expr = sys.argv[1].strip()
    try:
        pad = int(sys.argv[2])
    except (IndexError, ValueError):
        pad = 3

    if not expr:
        return

    parts = expr.split('..')
    try:
        start = int(parts[0])
        end = int(parts[1]) if len(parts) > 1 else start
        step = int(parts[2]) if len(parts) > 2 and parts[2] else 1
    except ValueError:
        sys.stderr.write(f"Invalid range expression: {expr}\n")
        sys.exit(1)

    if step == 0:
        sys.stderr.write("RUN range step must be non-zero\n")
        sys.exit(1)

    forward = start <= end
    step = abs(step) if forward else -abs(step)
    stop = end + (1 if forward else -1)
    
    # Generate run IDs
    ids = [f"{value:0{pad}d}" for value in range(start, stop, step)]
    print(" ".join(ids), end="")

if __name__ == "__main__":
    main()

