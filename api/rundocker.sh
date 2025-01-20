#!/bin/bash

# this is the install script 
#  install_script = "/opt/cosf/api/rundocker.sh"
# called on boot.

# this is the refresh script called from ssm for a refresh
#  #refresh_script = "/opt/cosf/api/docker-boot.sh" 

# file not found
#
pwd
ls -latr
. ./.env # for secrets
set -e # stop  on any error
#export ROOT="" # empty
export WORKSOURCE="/opt/cosf/api"

adduser --disabled-password --gecos "" cosf --home "/home/cosf"  || echo ignore
git config --global --add safe.directory "/opt/cosf"
git config --global --add safe.directory "/opt/cosf-memory"

cd "/opt/cosf/" || exit 1 # "we need cosf"
git log -2 --patch | head  -1000

mkdir -p "/var/cosf/agent_workspace/"
mkdir -p "/home/cosf"


cd "/opt/cosf/" || exit 1 # "we need cosf"

mkdir -p "/var/cosf/logs"
chown -R cosf:cosf "/var/cosf/" "/home/cosf" "/opt/cosf"

#if [ -f "/var/cosf/agent_workspace/boot_fast.sh" ];
#then
#    chmod +x "/var/cosf/agent_workspace/boot_fast.sh" || echo faild
    
#    # user install but do not start
#    su -c "bash -e -x /var/cosf/agent_workspace/boot_fast.sh" cosf
#fi
cd "/opt/cosf/" || exit 1 # "we need cosf"

mkdir -p "/var/run/cosf/secrets/"
mkdir -p "/home/cosf/.cache/huggingface/hub"

set +x
OPENAI_KEY=$(aws ssm get-parameter     --name "cosf_openai_key" | jq .Parameter.Value -r )
export OPENAI_KEY
echo "OPENAI_KEY=${OPENAI_KEY}" > "/var/run/cosf/secrets/env"
set -x

## append new homedir
# check if the entry exists already before appending pls
if ! grep -q "HF_HOME" "/var/run/cosf/secrets/env"; then
       echo "HF_HOME=/home/cosf/.cache/huggingface/hub" >> "/var/run/cosf/secrets/env"
fi

if ! grep -q "^HOME" "/var/run/cosf/secrets/env"; then
    echo "HOME=/home/cosf" >> "/var/run/cosf/secrets/env"
fi

if ! grep -q "^HOME" "/var/run/cosf/secrets/env"; then
# attempt to move the workspace
    echo "WORKSPACE_DIR=\${STATE_DIRECTORY}" >> "/var/run/cosf/secrets/env"
fi

# setup the systemd service again
cp "${WORKSOURCE}/nginx/site.conf" /etc/nginx/sites-enabled/default
cp "${WORKSOURCE}/systemd/cosf-docker.service" /etc/systemd/system/cosf-docker.service 
grep . -h -n /etc/systemd/system/cosf-docker.service

chown -R cosf:cosf /var/run/cosf/
mkdir -p /opt/cosf/api/agent_workspace/try_except_wrapper/
chown -R cosf:cosf /opt/cosf/api/


# always reload
# might be leftover on the ami,
systemctl stop swarms-uvicorn || echo ok
systemctl disable swarms-uvicorn || echo ok
rm /etc/systemd/system/swarms-uvicorn.service

systemctl daemon-reload
systemctl start cosf-docker || journalctl -xeu cosf-docker
systemctl enable cosf-docker || journalctl -xeu cosf-docker
systemctl enable nginx
systemctl start nginx

journalctl -xeu cosf-docker | tail -200 || echo oops
systemctl status cosf-docker || echo oops2

# now after cosf is up, we restart nginx
HOST="localhost"
PORT=8000
while ! nc -z $HOST $PORT; do
  sleep 1
  echo -n "."
done
echo "Port ${PORT} is now open!"

systemctl restart nginx
