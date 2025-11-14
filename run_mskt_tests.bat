@echo off
setlocal

rem 1) test_single_state.py
py -m pytest -q src\mskt_hmm\tests\tests_MsktHMM\test_single_state.py
if errorlevel 1 goto :fail

rem 2) test_single_state_MsktHMM.py
py -m pytest -q src\mskt_hmm\tests\tests_MsktHMM\test_single_state_equivalence_EMMIXskew.py
if errorlevel 1 goto :fail

rem 3) test_multi_state.py (con logging su file)
py -m pytest -qq ^
  --log-file="C:\opt\workspace\python\MsktHMM\src\mskt_hmm\tests\tests_MsktHMM\log\mskt_hmm_test.log" ^
  --log-file-level=INFO ^
  src\mskt_hmm\tests\tests_MsktHMM\test_multi_state.py
if errorlevel 1 goto :fail

echo All tests passed.
exit /b 0

:fail
echo One or more test runs failed.
exit /b 1
