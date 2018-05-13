#!/bin/csh

set org=$argv[1]
@ trainn=$argv[2]

ls list/$org*

foreach i ($org)
	echo $trainn
	@ wavn=`ls data/wav/$i | wc -l`
	@ tmp=$trainn + 1
	if ($wavn<$tmp) then
		echo "Number of wav is less than its of train"
		exit(1)
	endif
	sed -i $tmp,${wavn}d list/${i}_train.list
	sed -i 1,${trainn}d list/${i}_eval.list
end
