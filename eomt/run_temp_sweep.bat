@echo off
set ckpt=checkpoints/eomt_cityscapes.bin
set input1=../dataset/anomaly/RoadAnomaly21/images/*.*
set input2=../dataset/anomaly/RoadAnomaly/images/*.*
set input3=../dataset/anomaly/RoadObsticle21/images/*.*
set input4=../dataset/anomaly/fs_static/images/*.*
set input5=../dataset/anomaly/FS_LostFound_full/images/*.*
set method1=msp
set method2=maxlogit
set method3=maxentropy
set method4=rba

for %%t in (0.5 0.75 1.0 1.1 2.0) do (
  python evalAnomaly.py --ckpt %ckpt% --input %input1% --method %method1% --temp %%t
)

for %%t in (0.5 0.75 1.0 1.1 2.0) do (
  python evalAnomaly.py --ckpt %ckpt% --input %input1% --method %method2% --temp %%t
)

for %%t in (0.5 0.75 1.0 1.1 2.0) do (
  python evalAnomaly.py --ckpt %ckpt% --input %input1% --method %method3% --temp %%t
)

for %%t in (0.5 0.75 1.0 1.1 2.0) do (
  python evalAnomaly.py --ckpt %ckpt% --input %input1% --method %method4% --temp %%t
)


@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input2% --method %method1% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input2% --method %method2% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input2% --method %method3% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input2% --method %method4% --temp %%t
@REM )


@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input3% --method %method1% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input3% --method %method2% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input3% --method %method3% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input3% --method %method4% --temp %%t
@REM )



@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input4% --method %method1% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input4% --method %method2% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input4% --method %method3% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input4% --method %method4% --temp %%t
@REM )


@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input5% --method %method1% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input5% --method %method2% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input5% --method %method3% --temp %%t
@REM )

@REM for %%t in (0.5 0.75 1.0 1.1 2.0) do (
@REM   python evalAnomaly.py --ckpt %ckpt% --input %input5% --method %method4% --temp %%t
@REM )