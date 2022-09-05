# TODO

- [ ] Upper and lower bound initialization functions
- [ ] Bound initialization in sample?
- [ ] Bound updates in main code
- [ ] Test prune, sample, backup


# PLAN

- [ ] Tyler debugs tree_backup
- [ ] Ben debugs sample (Apparently this may not have been broken)
  - Only terminates because of ad hoc `max_steps` termination condition not present in actual algorithm. Still needs debugging.
- [ ] Ben starts tree_backup debug
- [ ] Tyler and Ben debug sample + tree_backup


# DEADLINE

- SARSOP.jl fully running by 1 Oct and providing reasonable outputs
- SARSOP.jl benchmarked with SARSOP cpp by 15 Oct
