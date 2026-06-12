export PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src"


freq_list=(
    "1d"
    "3d"
    "5d"
    "7d"
    "10d"
    "20d"
)

topk=(
    5
    10
    20
    30
    50
)

n_drop=(
    1
    2
    3
    5
    10
)

account_list=(
    10000
    100000
    1000000
)

for freq in "${freq_list[@]}"
do
    for topk in "${topk[@]}"
    do
        for n in "${n_drop[@]}"
        do
            for account in "${account_list[@]}"
            do
                echo "====================================================================================="
                echo "freq: $freq, topk: $topk, n_drop: $n, account: $account"
                echo "====================================================================================="
                python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
                    --account $account --freq "$freq" --topk "$topk" --n-drop "$n" \
                    --no-save
                echo "====================================================================================="
                echo ""
                echo ""
                echo ""
            done
        done
    done
done
