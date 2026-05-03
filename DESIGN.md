mergedog is a tool for making it easier to land OSS PR contributions to
pytorch/pytorch.  The primary design goals:

- **Adoption.** Once we have acknowledge a PR as good, we take full control
  over the rest of the process of shepherding the PR in.  The original author
  is not involved.

- **Security.** In fact, it is actively harmful if the external contributor
  intervenes after we start the shepherding process, as they can introduce
  unsafe code after approval.  Interventions from an untrusted user are ignored,
  or, in the worst case, this halts the merge process and we have to take over.

- **Autonomous.** I want to ask for a PR to be merged, and then I don't want
  to touch it until it's done.

- **Synchronous.** The shepherd process happens in the *foreground* of a
  terminal process.  One process per PR to shepherd.  You're welcome to write a
  muxer on top, but this design makes it simple.

Here is the workflow we implemetn:

1. We mark a PR as approved.  The exact commit which we approved indicates trust.
   We cannot land an untrusted commit.

2. We get CI to be green.  To have been approved, this indicates that we trust an
   LLM to solve any pending CI problems.  This is the bulk of the autonomous process;
   mergedog:

   a. Approves the CI run
   b. Polls for CI results
   c. Decides if the CI results indicate real failures, and if they do, add a commit
      with the changes, clearly annotated with [MERGEDOG] in the commit title.

   We assume any CI failures can be addressed without having to test locally.
   If mergedog fails to successfully one-shot a CI failure, this means it is too
   complicated and we should quit out for human intervention.

   When plain CI is passing, mergedog applies ciflow/trunk and same applies.

3. We wait for a human to review the mergedog and manually trigger a pytorchbot merge.

mergedog does NOT work on ghstack diff stacks.

mergedog is implemented as a traditional software harness that shells into Claude/Codex to
actually issue the fixes.
