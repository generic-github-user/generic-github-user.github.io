{{ site_header }}

I run [NixOS](https://nixos.org/) on a Lenovo ThinkPad T590 with a 500gb NVMe SSD and 32gb of RAM. I use a ZFS-based [impermanence](https://wiki.nixos.org/wiki/Impermanence) setup to regularly erase my root directory[^1] and `/var`; only `/home`, `/nix`, and a handful of system state directories (bind-mounted to a persistent dataset) are retained on boot. My main desktop environment is a thoroughly un-riced [Sway WM](https://swaywm.org/).

My terminal emulator of choice is [kitty](https://sw.kovidgoyal.net/kitty/), though I still use [konsole](https://apps.kde.org/konsole/) from time to time. I use [tmux](https://github.com/tmux/tmux/wiki) with [mosh](https://mosh.org/) for persistent SSH sessions into remote machines. My server and other laptops also run NixOS. For development, I mainly use [neovim](https://neovim.io/) with very few plugins or modifications.

[^1] This may sound extreme, but it is essentially the only way to regain control over your system state and be confident you are storing/backing up exactly what you need (and no more) in a world rife with software that does not respect user data, cleanliness, or separation of concerns. Moreover (and as is more commonly advertised), it makes it much easier to ~guarantee reproducibility of a system, which is surprisingly pleasant to have even for personal laptops and the like.

This site is generated statically from Markdown files using [jinja](https://jinja.palletsprojects.com/en/stable/) and [pandoc](https://pandoc.org/), and served using [GitHub Pages](https://docs.github.com/en/pages). It is designed to be (at least in part) a wiki-style blog in which posts are updated and reorganized at my discretion, with no particular posting schedule or guarantees of completeness.
