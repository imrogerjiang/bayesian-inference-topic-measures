sim_name=sig_n_raters

python3 create_sim_settings.py --n_raters "(15, 70)" --scores_per_r 38 --n_sims 1000 --sim_name $sim_name

gnome-terminal \
--tab --title="0" --command="bash -c 'python3 perform_sig_test.py --process 0 --sim_name $sim_name; $SHELL'" \
--tab --title="1" --command="bash -c 'python3 perform_sig_test.py --process 1 --sim_name $sim_name; $SHELL'" \
--tab --title="2" --command="bash -c 'python3 perform_sig_test.py --process 2 --sim_name $sim_name; $SHELL'" \
--tab --title="3" --command="bash -c 'python3 perform_sig_test.py --process 3 --sim_name $sim_name; $SHELL'" \
--tab --title="4" --command="bash -c 'python3 perform_sig_test.py --process 4 --sim_name $sim_name; $SHELL'" \
--tab --title="5" --command="bash -c 'python3 perform_sig_test.py --process 5 --sim_name $sim_name; $SHELL'"