#!/bin/sh
set -e
USER=50000

chown $USER:$USER /plugin

if [[ "${DEV_ENV:-false}" == "true" ]] ; then
    cd /plugin

    if [ -d ".git" ]; then
       git pull
    else
       rm -f -r *
       git clone $github_repo . --depth=5
    fi

    #on commit ignore the permission changes
    git config core.filemode false

    chmod 777 /plugin
    chmod -R 775 /plugin/*
else
    rm -rf /llm_plugin/*
    cp /plugin -R /llm_plugin/
fi
