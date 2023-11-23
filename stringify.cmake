file(READ ${input_file} content)
set(delim "====")
set(content "R\"${delim}(\n${content})${delim}\"")
file(WRITE ${output_file} "${content}")