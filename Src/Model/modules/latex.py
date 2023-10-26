import re

# Embed latex by putting it within begin-end statements of different environment types
def embed_in_environment(content, environment_name, params_string  = ""):
    result = "\\begin{" + environment_name + "}" + params_string + "\n"
    result += content
    result += "\\end{" + environment_name + "}\n"
    return result


# Make the dataframe object ready to be converted into latex format.
# Includes: 
#       - changing column names
#       - replacing zeros and ones with symbols of inclusion or exclusion
#       - rounding error values to 4 decimals
#       - make lowest values in column bolded
def prepare_df_for_latex(df_org):
    df = df_org.copy()
    start_feat, start_err = 1, 6
    
    # Rename columns    
    df.columns = [s.capitalize() for s in df.columns[:start_err]] + [s.upper() for s in df.columns[start_err:]]
    df.rename({"Ti":"Techn. Ind.", "Combination_number" : "Nr."}, axis = 1, inplace = True)

    data_cols = df.columns[start_feat:start_err]
    error_cols = df.columns[start_err:]
    
    # Replace zeros and ones with x and circle; round up error values
    df[data_cols] = df[data_cols].applymap(lambda x: "$\\circ$" if x == 1 else "$\\times$")
    df[error_cols] = df[error_cols].applymap(lambda x: round(float(x), 4))

    # Make lowest values of errors bolded
    for c in error_cols:
        df.loc[df[c].idxmin(), c] = "\\textbf{" + str(df.loc[df[c].idxmin(), c]) + "}"

    return df


def repeat_column_style(col_type, col_size, rep_nr, border_between=False, end_line = True):
    res = "|"
    elements = [col_type + "{" + str(col_size) + "cm}" for i in range(rep_nr)]
    join_char = "|" if border_between == True else ""
    end_line_el = "|" if end_line else ""
    res += join_char.join(elements) + end_line_el
    return res
    
# Replace horizontal line formatting
# Add custom vertical borders and column definitions
def format_table_borders(latex_table):
    result_string = latex_table
    for rule in ["toprule", "bottomrule", "midrule"]:
        result_string = result_string.replace(rule, "hline")

    column_def = repeat_column_style("P", 0.7, 1, end_line=False) + repeat_column_style("P", 1.3, 5, end_line=False) + repeat_column_style("P", 1.1, 4)
    result_string = re.sub(r'begin{tabular}\{[^}]*\}', "begin{tabular}{" + column_def + "}", result_string)
    return result_string

# Format data frame, convert it to latex, add formatting and save to file
def convert_and_save_df_to_latex(df, file_path, caption = "", label = ""):
    df_formatted = prepare_df_for_latex(df)
    tabular_tex = df_formatted.to_latex(index=False, escape=False)
    tabular_tex = format_table_borders(tabular_tex)

    tabular_tex = "\\centering\n" + tabular_tex + "\\caption{"+ caption + "}\n" + "\\label{tab:"+ label + "}\n"
    table_tex = embed_in_environment(tabular_tex, environment_name="table", params_string="[H]")

    with open(file_path, 'w') as file:
        file.write(table_tex)

    