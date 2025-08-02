#!/bin/bash
echo starting chat web server
cd ${APP_PATH}/buttermilk/frontend/chat
npm i . && npm run dev ${FRONTEND_PARAMS}
