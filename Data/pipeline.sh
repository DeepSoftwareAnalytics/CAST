cd ../source_code/datapreprocessed/
mkdir -p log
### step 1 generating java files
python  gen_java_files.py |tee ./log/1_gen_java_files.txt

### step 2 generating splitted AST using progex_s
mkdir -p ../../Data/TL_CodeSum/splitted_ast
java -jar progex_s.jar -outdir  ../../Data/TL_CodeSum/splitted_ast  -ast -lang java -format dot  -node_level block   ../../Data/TL_CodeSum/java_files

### step 3 getting correct splitted ast fid
python get_correct_splitted_ast.py |tee ./log/3_get_correct_splitted_ast.txt

## step 4.1 get_splitted_ast.py
python get_splitted_ast.py | tee ./log/4_get_splitted_ast.txt

## step 4.2 get_splitted_subtoken_ast.py
python get_splitted_ast_subtoken_node.py | tee ./log/4_get_splitted_ast.txt

## step 5 get_flatten_ast.py
python get_flatten_ast.py  |tee ./log/5_1_get_flatten_ast.txt

## step 5_1_get_big_graph_fid.py
python get_big_graph_fid.py  |tee ./log/5_2_get_big_graph_fid.txt

## step 6 get_rebuild_tree.py
python get_rebuild_tree.py |tee ./log/6_get_rebuild_tree.txt

## step 7.1 code an summary data_processing
python code_summary_processing.py  |tee ./log/7_1_code_summary_processing.txt

## step 7.2filter_summary.py
python write_code_summary.py  |tee ./log/7_2_write_code_summary.txt

## step 8.1 code an summary data_processing
python ast_subtoken_word_count.py  |tee ./log/8_1_ast_subtoken_word_count.txt

## step 8.2filter_summary.py
python write_asts.py  |tee ./log/8_2_write_asts.txt

