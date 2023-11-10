
abs_dir_path=$(realpath $(dirname $1))
if [ ! -z "${TMPDIR}" ]; then
  # TMPDIR interferes with mktemp so we need to hide it
  tmpdir_backup=$TMPDIR
  unset TMPDIR
fi
script_path=$(mktemp -p ${abs_dir_path} -t XXXXXX.py)
if [ ! -z "${tmpdir_backup}" ]; then
  # TMPDIR was unset and we need to restore it
  TMPDIR=$tmpdir_backup
fi

# FIXME if there's a shebang in script_path, it won't be in the first line anymore
echo "import torch_performance_linter" > ${script_path}
cat $1 >> ${script_path}

PYTORCH_JIT=0 python ${script_path} ${@:2} &
pid=$!
wait $pid

echo $TPL_MODE
echo "--------------------------------- SUMMARY ---------------------------------"
if [ "${TPL_MODE}" = "TIMER" ]; then
  cat torch_performance_linter_timer_${pid}.txt
else
  cat torch_performance_linter_summary_${pid}.txt
fi
echo ""
echo "---------------------------------------------------------------------------"
echo "See full log in torch_performance_linter_${pid}.log"

rm ${script_path}
