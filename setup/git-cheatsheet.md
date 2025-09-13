# Git Cheatsheet
Cheatsheet of most important git operations from command line. For first-time setup, see the **Cloning** section at the end. Note that the "version" on **GitHub** is called the **remote** repository, whereas you work directly with the **local** repository.

## Commit, Push & Pull
Git is a software for "version control". It does this by tracking changes to files using **commits** (groupings of changes), which can be **pushed** from local to remote, or **pulled** from remote to local. The push workflow looks as follows:
1) ```git status``` [optional] will show what files have changed
2) ```git add file1.py file2.txt``` will "add" the file for the next  **commit** (also known as **staging**)
3) ```git commit -m "Describe change here``` will create a local **commit**
4) ```git push``` will **push** the commit you just made to the remote repository

Now to get any remote changes (changes by other users) to your local repository, use ```git pull```.
<br>

Note that any changes that have been **added**, or **commited** that has not been **pushed** will cause a conflict on the next **pull**, and if there are any new **commits** others have **pushed** to the remote repository, and you have local **added** or **commited** changes, the next **pull** will also cause a conflict. To resolve this, you can **rebase**.
```sh
git rebase
```

A rebase will essentially take all you **commits**, "undo them" temporarily, **pull** the remote commits, then re-apply your changes (this is important for Git as the order of changes matters).

## Cloning
This is the process of "downloading" the GitHub repository (remote repository), while retaining the ability to **push** and **pull** changes to/from it.

### SSH Key Generation
Ensure that you have the appropriate SSH keys for using GitHub. Ensure that you remember in which directory the files are created (typically `C:/Users/xxx/.ssh/` for Windows or `~/.ssh/` for Linux).
```sh
ssh-keygen -t ed25519
```

Then copy the contents of the `id_ed25519.pub` file, and go to your GitHub accounts `settings > SSH and GPG keys` (*SSH and GPG keys* are under the *Access* grouping in the settings page), and press `New SSH key`. Give it an appropriate *Title* and paste the contents of the `id_ed25519.pub` file there (leave the *Key type* as *Authentication Key*). For linux, you can quickly see the contents of this file in terminal using:
```sh
cat ~/.ssh/id_ed25519.pub # Copy the entire output
```

### Git Config
In order to clone a GitHub repository, there are 2 key config items. The first is your **name**. GitHub will accept p. much anything, but use something recognizable (like your full name). The second is you **email**. This must be tied to your GitHub account, otherwise cloning repositories will not work.
```sh
git config --global user.name "Full Name"
git config --global user.email "github.account.email@domain.com
```

### Git Clone
Now you can clone the GitHub repository. The *target* of this command is standardized for GitHub repositories as `git@github.com:USER/REPOSITORY.git`, where `USER` is the username for the account hosing the repository. For this repository, the `USER` is `Ferdi0412`, and the `REPOSITORY` is `traffic-signal-control`.
```sh
git clong git@github.com:Ferdi0412/traffic-signal-control
```