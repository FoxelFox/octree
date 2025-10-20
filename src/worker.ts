import {Request, Result} from "./generation/message";

import init, {add} from "src/my-lib/pkg"

let initialized = false;

onmessage = async (e: MessageEvent<Request>) => {


    if (!initialized) {
        await init();
        initialized = true;
    }

    let result: Result = {
        id: e.data.id
    }


    console.log("TEST")

    switch (e.data.operation) {
        case 'add':
            result.data = add(e.data.args[0], e.data.args[1]);
            //result.data = e.data.args[0] + e.data.args[1];
            break;
    }


    setTimeout(() => {
        postMessage(result);
    }, 1000);
};