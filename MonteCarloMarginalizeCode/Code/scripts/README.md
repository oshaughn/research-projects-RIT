


# Creating JS values for all pairs


```
  # create command lines for JS values between all pairs of filenames, then execute those commands
  for param in mc q xi; do 
     ls fulltest_osg_jared_aligned*/G336694*/posterior_samples-4.dat | ./tool_pairs.py --prefix="convergence_test_samples.py --method JS --parameter ${param}" > tmpfile; 
    ./tmpfile > jsvals_G336694_${param}_jared_it4; 
   done
  # create mean, std for them
  for i in mc q xi; do echo $i `python tool_mean_std.py jsvals_G336694_${i}_jared_it4`; done

```
