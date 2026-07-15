import re
import pandas as pd
import sympy as sp


def split_latex_expression(latex_str, max_len):
    """
    Splits a LaTeX math string into multiple aligned rows at top-level '+' or '-'
    operators if it exceeds max_len.
    """
    # Strip any outer math mode delimiters
    was_wrapped = False
    if latex_str.startswith('$') and latex_str.endswith('$'):
        latex_str = latex_str[1:-1]
        was_wrapped = True

    parts = []
    current_part = []
    brace_level, paren_level, bracket_level = 0, 0, 0

    # Identify top-level operators for potential splits
    for char in latex_str:
        if char == '{':
            brace_level += 1
        elif char == '}':
            brace_level -= 1
        elif char == '(':
            paren_level += 1
        elif char == ')':
            paren_level -= 1
        elif char == '[':
            bracket_level += 1
        elif char == ']':
            bracket_level -= 1

        # Split immediately before '+' or '-' at the root level
        if brace_level == 0 and paren_level == 0 and bracket_level == 0 and char in ('+', '-'):
            if current_part:
                parts.append("".join(current_part).strip())
            current_part = [char]
        else:
            current_part.append(char)

    if current_part:
        parts.append("".join(current_part).strip())

    if not parts or len(parts) == 1:
        return f"${latex_str}$" if was_wrapped else latex_str

    # Rebuild segments to fit within max_len
    lines = []
    current_line = parts[0]

    for part in parts[1:]:
        if len(current_line) + len(part) > max_len:
            lines.append(current_line)
            # Prepend the alignment operator '&' for multi-line formatting
            current_line = f"& {part}" if (part.startswith('+') or part.startswith('-')) else f"& {part}"
        else:
            current_line += f" {part}"
    lines.append(current_line)

    # If the expression was split, wrap it in an amsmath aligned environment
    if len(lines) > 1:
        formatted_lines = [lines[0]] + [l if l.startswith('&') else f"& {l}" for l in lines[1:]]
        res = "\\begin{aligned}[t] " + " \\\\ ".join(formatted_lines) + " \\end{aligned}"
    else:
        res = lines[0]

    return f"${res}$" if was_wrapped else res


def format_coefficient(val):
    """
    Formats a float value according to the custom rules:
    - If absolute value is >= 0.1, round to 2 decimal places.
    - If absolute value is < 0.1, use 2-decimal scientific notation (e.g., 1.23E-3).
    """
    abs_val = abs(val)
    if abs_val == 0:
        return "0"

    if abs_val >= 0.1:
        formatted = f"{val:.2f}"
        if formatted.endswith(".00"):
            formatted = formatted[:-3]
        return formatted
    else:
        formatted = f"{val:.2e}".upper()
        # Clean up the exponent (remove leading zeros)
        if "E-" in formatted:
            base, exp = formatted.split("E-")
            formatted = f"{base}E-{int(exp)}"
        elif "E+" in formatted:
            base, exp = formatted.split("E+")
            formatted = f"{base}E+{int(exp)}"
        return formatted


def customize_expr_numbers(expr):
    """
    Traverses the SymPy expression and replaces all Float instances
    with a Symbol holding our custom-formatted string representation.
    """
    if isinstance(expr, sp.Float):
        return sp.Symbol(format_coefficient(float(expr)))
    elif expr.args:
        new_args = [customize_expr_numbers(arg) for arg in expr.args]
        return expr.func(*new_args)
    return expr


def post_process_latex_scientific(latex_str):
    """
    Finds instances of scientific notation like 1.23E-3 or 4.50E-5 in the
    LaTeX string and wraps the 'E' part in \text{} to conserve table space.
    """
    # Matches a number, followed by 'E', followed by a sign and digits
    pattern = r"(\d+(?:\.\d+)?)E([+-]?\d+)"
    # Replaces it with: [number]\text{E[exponent]}
    return re.sub(pattern, r"\1\\text{E\2}", latex_str)


def df_to_latex_double_column(parquet_path, difficulty_char, output_tex_path):
    df = pd.read_parquet(parquet_path)
    if len(df) < 50:
        print(f"Warning: {parquet_path} only has {len(df)} rows. Table will be padded.")

    def to_math(expr_str, is_skeleton=False):
        try:
            expr = sp.parse_expr(expr_str)
            if not is_skeleton:
                expr = customize_expr_numbers(expr)

            raw_latex = sp.latex(expr)
            # Post-process scientific notation to inject \text{E...}
            processed_latex = post_process_latex_scientific(raw_latex)
            return f"${processed_latex}$"
        except Exception:
            return f"\\verb|{expr_str}|"

    df['sk_latex'] = df['skeleton'].apply(lambda x: to_math(x, is_skeleton=True))
    df['eq_latex'] = df['equation'].apply(lambda x: to_math(x, is_skeleton=False))

    left_half = df.iloc[:25].reset_index(drop=True)
    right_half = df.iloc[25:50].reset_index(drop=True)

    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{Curated Equations for {difficulty_char.upper()} Complexity}}",
        "\\label{tab:eq_" + difficulty_char.lower() + "}",
        "\\small",
        "\\begin{tabular}{cll}",
        "\\hline\\hline",
        "\\textbf{ID} & \\textbf{Skeleton} & \\textbf{Equation} \\\\",
        "\\hline"
    ]

    for idx, row in df.iterrows():
        eq_id = f"{difficulty_char.upper()}{idx + 1}"
        sk_latex = row['sk_latex']
        eq_latex = row['eq_latex']
        row_str = f"{eq_id} & {sk_latex} & {eq_latex} \\\\"
        latex_lines.append(row_str)

    latex_lines.extend([
        "\\hline\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    with open(output_tex_path, "w") as f:
        f.write("\n".join(latex_lines))
    print(f"✓ Saved space-optimized LaTeX table to: {output_tex_path}")


# Generate all three LaTeX tables
df_to_latex_double_column("../data/curated_datasets_easy.parquet", "E", "table_easy.tex")
df_to_latex_double_column("../data/curated_datasets_medium.parquet", "M", "table_medium.tex")
df_to_latex_double_column("../data/curated_datasets_hard.parquet", "H", "table_hard.tex")