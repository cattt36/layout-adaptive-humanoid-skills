#!/bin/bash
# Pack the final v2-pipeline artifacts (no raw rollouts) for the local project folder.
H=/root/gpufree-data/hnu_v2
cd $H
L=pack_list.txt
: > $L
echo data/ref_v1 >> $L
echo data/annot_v2 >> $L
echo data/z_tables_v3 >> $L
echo data/z_tables >> $L
echo data/heuristic >> $L
echo runs/latent_v3/seed42 >> $L
echo runs/latent_v2/seed42 >> $L
ls runs/students_v3/*/student_2500.pt runs/students_v3/*/args.json runs/students_v3/*/train_log.jsonl >> $L
ls runs/students/*/student_2500.pt runs/students/*/args.json runs/students/*/train_log.jsonl >> $L
ls runs/quality_v3/*/quality_net.pt runs/quality_v3/*/quality_eval.json runs/quality/*/quality_eval.json >> $L
ls runs/expert_eval/*/*.json runs/expert_eval/*/checkpoint.txt >> $L
ls runs/results/*.json runs/results_v3/*.json >> $L
for t in ThreeStepPoleRun FiveStepPoleRun CrouchedTraverse; do
  c=$(cat runs/expert_eval/$t/checkpoint.txt); echo ${c#$H/} >> $L; echo $(dirname ${c#$H/})/params >> $L
done
echo paper_root/offline_81_summary.md >> $L
for d in paper_root/*/Results/8.1_Offline_Representation/outputs; do echo $d >> $L; done
tar czf /root/gpufree-data/hnu_v2_outputs.tgz -T $L
ls -la /root/gpufree-data/hnu_v2_outputs.tgz
