#!/bin/bash
echo starting chat web server
cd buttermilk/frontend/chat
npm i . && npm run dev ${FRONTEND_PARAMS}
