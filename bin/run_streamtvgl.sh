# dir setting
cd `dirname $0`
cd ..
cd exp

python3 train_streamtvgl.py --dims 5 5  \
    --breaks-json '[[0, 100, 200, 300],[0, 100, 200, 300]]' \
    --seeds-json '[[1,2,3],[4,5,6]]' \
    --T 300 \
    --window-size 50 \
    --step-size 1