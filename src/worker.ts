import {Request, Result} from "./generation/message";

import init, {noise_for_chunk} from "src/my-lib/pkg"

let initialized = false;

onmessage = async (e: MessageEvent<Request>) => {

    const start = performance.now();

    if (!initialized) {
        await init();
        initialized = true;
    }

    let result: Result = {
        id: e.data.id
    }


    console.log("TEST")

    switch (e.data.operation) {
        case 'noise_for_chunk':
            result.data = noise_for_chunk(e.data.args[0], e.data.args[1], e.data.args[2], e.data.args[3]);
            //result.data = e.data.args[0] + e.data.args[1];
            break;
    }


    console.log(performance.now() - start)
    postMessage(result);
};