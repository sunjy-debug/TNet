
if [ $1 -eq 1 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=BC --expID=$id --flipRate=1 --lr_1step=0.0001 --lr_2step=0.01 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.1 --alpha=0.5 --gamma=1.0
    done
fi

if [ $1 -eq 2 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=BC_hete --expID=$id --flipRate=1 --lr_1step=0.001 --lr_2step=0.0001 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.05 --alpha=0.5 --gamma=1.0
    done
fi

if [ $1 -eq 3 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=Flickr --expID=$id --flipRate=1 --lr_1step=0.0001 --lr_2step=0.0001 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.25 --alpha=0.5 --gamma=0.5
    done
fi

if [ $1 -eq 4 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=Flickr_hete --expID=$id --flipRate=1 --lr_1step=0.001 --lr_2step=0.001 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.2 --alpha=0.5 --gamma=1.0
    done
fi


if [ $1 -eq 5 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=BC_hete_z --expID=$id --flipRate=1 --lr_1step=0.001 --lr_2step=0.001 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.2 --alpha=1. --gamma=1.0
    done
fi

if [ $1 -eq 6 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=Flickr_hete_z --expID=$id --flipRate=1 --lr_1step=0.001 --lr_2step=0.001 --num_grid=20 --beta=20 --epochs=160 --tr_knots=0.2 --alpha=1. --gamma=1.0
    done
fi

if [ $1 -eq 7 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=TargetedModel_DoubleBSpline --dataset=Simulation --n_nodes=3000 --edge_prob=0.2 --expID=0 --flipRate=1 --lr_1step=0.001 --lr_2step=0.0001 --num_grid=20 --beta=20 --epochs=20 --tr_knots=0.2 --alpha=2. --gamma=2.0
    done
fi

if [ $1 -eq 8 ]; then
    for id in {0,1,2,3,4} ; do
        CUDA_VISIBLE_DEVICES=1 python main.py --model=NetEsimator --dataset=Simulation --n_nodes=3000 --edge_prob=0.2 --lrD=0.001 --lrD_z=0.001 --lr=0.001 --weight_decay=0.001 --epochs=20 --dstep=50 --d_zstep=50 --pstep=1 --alpha=2. --gamma=2.0
    done
fi