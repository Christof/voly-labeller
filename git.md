# Git

The configured mail address must be one of the addresses set in the github account, so that
github can associate all commits to an user.

**Development of new features** and bug-fixes must be done on branches:
```
git checkout -b <branch-name>
gco -b <branch-name>
```

All samples show first the full git command and then the short version using our aliases and custom commands.

Files can be **added** to the staging area with:
```
git add <file-name or pattern>
ga <file-name or pattern>
```

A **commit** can be created with:
```
git commit -m "<message>"
gc -m "<message>"
```

- Use imperative, present tense for commit messages.
- Start with an uppercase letter and end with punctuation.
- Lines in the message must not be longer than 72 characters.
- The first line should be like an email subject, short but to the point.
- Use github keywords to close or link issues ([see](https://help.github.com/articles/closing-issues-via-commit-messages/)).

The work can be **pushed** (for the first time) to github using:
```
git push --set-upstream origin <remote-branch-name>
git push -u origin <remote-branch-name>
gpu
```
With this command the local branch tracks the created remote branch.
**Subsequent pushes** can be done with:
```
git push
gp
```
This should be done at least before leaving the working place, to have a secure backup of the work.

**Pulling changes** from github can be done with:
```
git pull
gl
```

If the **branch is finished**, a pull request (PR) on github must be created. Before creating the PR changes
which have been made in the meantime on `master` (or the merge target branch) should be rebased into
the branch.
```
git checkout master
gco master

git pull
gl

git checkout <branch-name>
gco <branch-name>

git rebase master
grb master

```

Resolve and merge conflicts manually:

* check for conflicts - conflict exist if: "both modified" appears:
```
git status
```
* open file and edit/resolve conflicts. If you do not want to edit the file use:
git checkout --theirs/ours <conflicted file>

* add the modified file and continue with the next conflict:
```
git add <modified file>
git rebase --continue 
(grb --continue)
```


When finished, push your changes:
```
git push --force
gp -f
```

Mind that a force push overwrites the branch on github.

Now a PR on github can be created. After the build server has confirmed that the new changes compile, don't break any tests and adhere to the style guidelines, another team member must be asked to review
the changes and merge the PR.

Finally the local branch can be deleted using:
```
git branch -d <branch-name>
gb -d <branch-name>
```

Rinse and :repeat: :smiley:

## Blog posts
[How to write the perfect pull request](https://github.com/blog/1943-how-to-write-the-perfect-pull-request)
