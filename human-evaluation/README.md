## Human Evaluation

### Approaches

* Recent SOTA approaches: `NCS`.
* Different AST representation approaches:  `Astattgru`, `CodeASTNN`. 
* ur approach: `CAST`

### Questionnaires 

We randomly choose 50 Java methods from the testing sets 
(25 from TL-CodeSum and 25 from Funcom) and divide them into five groups.Each volunteer is asked to assign scores from 0 to 4 (the higher the better) 
to the generated summary from the three aspects:
* Similarity: similarity of the generated summary and the ground truth summary
* Naturalness: grammaticality and fluency
* Informativeness: the amount of content carried over from the input code to the generated summary, ignoring fluency.

Human eval 1- 10 https://www.wjx.cn/vj/Q00Al4G.aspx 

human eval 21-30 https://www.wjx.cn/vj/ekf2NWM.aspx

human eval 21-30 https://www.wjx.cn/vj/ekf2NWM.aspx 

human eval 31-40 https://www.wjx.cn/vj/YLDdXyZ.aspx 

human eval 41-50 https://www.wjx.cn/vj/h4DscfE.aspx 

### Result & Analysis
 
Please run the following command:

    python human_eval.py 2>&1 | tee human_eval_result.log

You can get the following result. More detail can see [human_eval_result_log](human_eval_result_log)

informative

    +---------------+----+----+----+----+----+------------+-----+-----+----+
    |   model type  | 0  | 1  | 2  | 3  | 4  |  Avg(Std)  |  ≥3 |  ≥2 | ≤1 |
    +---------------+----+----+----+----+----+------------+-----+-----+----+
    |      CAST     | 12 | 32 | 32 | 45 | 79 | 2.74(1.29) | 124 | 156 | 44 |
    | Ast-attendgru | 11 | 34 | 72 | 59 | 24 | 2.26(1.05) |  83 | 155 | 45 |
    |      NCS      | 8  | 36 | 63 | 56 | 37 | 2.39(1.1)  |  93 | 156 | 44 |
    |   CodeASTNN   | 8  | 32 | 59 | 65 | 36 | 2.44(1.08) | 101 | 160 | 40 |
    +---------------+----+----+----+----+----+------------+-----+-----+----+

naturalness

    +---------------+----+----+----+----+-----+------------+-----+-----+----+
    |   model type  | 0  | 1  | 2  | 3  |  4  |  Avg(Std)  |  ≥3 |  ≥2 | ≤1 |
    +---------------+----+----+----+----+-----+------------+-----+-----+----+
    |      CAST     | 9  | 22 | 23 | 36 | 110 | 3.08(1.23) | 146 | 169 | 31 |
    | Ast-attendgru | 17 | 27 | 42 | 59 |  52 | 2.46(1.31) | 111 | 153 | 44 |
    |      NCS      | 8  | 24 | 47 | 45 |  76 | 2.78(1.19) | 121 | 168 | 32 |
    |   CodeASTNN   | 6  | 21 | 29 | 55 |  89 | 3.0(1.13)  | 144 | 173 | 27 |
    +---------------+----+----+----+----+-----+------------+-----+-----+----+

similarity

    +---------------+----+----+----+----+----+------------+-----+-----+----+
    |   model type  | 0  | 1  | 2  | 3  | 4  |  Avg(Std)  |  ≥3 |  ≥2 | ≤1 |
    +---------------+----+----+----+----+----+------------+-----+-----+----+
    |      CAST     | 16 | 27 | 37 | 50 | 70 | 2.66(1.29) | 120 | 157 | 43 |
    | Ast-attendgru | 18 | 43 | 75 | 45 | 19 | 2.02(1.09) |  64 | 139 | 61 |
    |      NCS      | 17 | 38 | 65 | 54 | 26 | 2.17(1.14) |  80 | 145 | 55 |
    |   CodeASTNN   | 18 | 33 | 67 | 55 | 27 | 2.2(1.14)  |  82 | 149 | 51 |
    +---------------+----+----+----+----+----+------------+-----+-----+----+