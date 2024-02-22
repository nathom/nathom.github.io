document.addEventListener("DOMContentLoaded", () => {
    // Get the canvas element
    let drawing = true;
    let isDrawing = false;
    let brushRadius = 2.8;
    const canvas = document.getElementById("drawingCanvas");
    const context = canvas.getContext("2d");

    const bgCanvas = document.getElementById("barGraphCanvas");
    const bgContext = bgCanvas.getContext("2d");

    const nnCanvas = document.getElementById("nnBarGraphCanvas");
    const nnContext = nnCanvas.getContext("2d");

    const convCanvas = document.getElementById("convBarGraphCanvas");
    const convContext = convCanvas.getContext("2d");

    const clearButton = document.getElementById("clearButton");
    const eraserToggle = document.getElementById("eraserToggle");
    const floatSlider = document.getElementById("floatSlider");
    const sliderLabel = document.getElementById("sliderLabel");

    const lsBgTitle = document.getElementById("ls-inference");
    const nnBgTitle = document.getElementById("nn-inference");
    const convBgTitle = document.getElementById("conv-inference");

    const weightsUrl =
        "https://raw.githubusercontent.com/nathom/ece174_mini_project/main/resources/weights.json";
    // const nnWeightsUrl =
    //     "https://raw.githubusercontent.com/nathom/ece174_mini_project/main/resources/weights_nn.json";
    const nnWeightsUrl =
        "https://raw.githubusercontent.com/nathom/ece174_mini_project/main/resources/weights_fcn_aug.json";
    const convWeightsUrl =
        "https://raw.githubusercontent.com/nathom/ece174_mini_project/main/resources/weights_conv_aug.json";
    // TODO: make async

    const weights = fetchWeights(weightsUrl);
    const nnWeights = fetchWeights(nnWeightsUrl);
    const convWeights = fetchWeights(convWeightsUrl);
    const pixels = new Array(28 * 28).fill(0);

    const dummyScore = new Array(10).fill(0);
    drawGrid(context, canvas, pixels);
    drawBarGraph(bgContext, bgCanvas, dummyScore);
    drawLogitsBarGraph(nnContext, nnCanvas, dummyScore);
    drawLogitsBarGraph(convContext, convCanvas, dummyScore);

    function startDrawing(e) {
        isDrawing = true;
        draw(e); // Start drawing immediately
    }

    /**
     * @param {number[]} pixels
     */
    async function stopDrawing(pixels) {
        isDrawing = false;
        const lsStart = performance.now();
        const score = evalLSModel(pixels, await weights);
        const lsTime = performance.now() - lsStart;
        const nnStart = performance.now();
        const nnScore = evalNN(pixels, await nnWeights);
        const nnTime = performance.now() - nnStart;
        const convStart = performance.now();
        const convScore = evalConv(pixels, await convWeights);
        const convTime = performance.now() - convStart;
        updatePerfStats(lsTime, nnTime, convTime);
        drawBarGraph(bgContext, bgCanvas, score);
        drawLogitsBarGraph(nnContext, nnCanvas, nnScore);
        drawLogitsBarGraph(convContext, convCanvas, convScore);
    }

    /**
     * @param {number} lsTime
     * @param {number} nnTime
     * @param {number} convTime
     */
    function updatePerfStats(lsTime, nnTime, convTime) {
        const ls = Math.round(lsTime);
        const nn = Math.round(nnTime);
        const conv = Math.round(convTime);
        lsBgTitle.textContent = `1v1 Least Squares (${ls} ms)`;
        nnBgTitle.textContent = `Fully Connected Network (${nn} ms)`;
        convBgTitle.textContent = `Convolutional Network (${conv} ms)`;
    }

    function draw(e) {
        if (!isDrawing) return;

        const { x, y, width } = canvas.getBoundingClientRect();
        const scale = canvas.width / width;
        const truex = (e.clientX - x) * scale;
        const truey = (e.clientY - y) * scale;

        // Set drawing styles
        fillPixel(context, canvas, drawing, pixels, truex, truey, brushRadius);
    }
    function updateBrushRadius() {
        const selectedValue = parseFloat(floatSlider.value);
        brushRadius = selectedValue;
        sliderLabel.textContent = `Brush radius: ${brushRadius.toFixed(1)}`;
    }
    /**
     * @param {Array<number>} pixels Array of pixels
     */
    function clearAllPixels(pixels) {
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                pixels[y * 28 + x] = 0;
            }
        }
        drawGrid(context, canvas, pixels);
    }
    /**
     * @param {HTMLElement} button
     */
    function toggleMode(button) {
        drawing = !drawing;
        if (drawing) {
            button.textContent = "Switch to Eraser";
            button.style.backgroundColor = "var(--fg0)";
            button.style.color = "var(--bg0)";
        } else {
            button.textContent = "Switch to Pencil";
            button.style.backgroundColor = "var(--bg0)";
            button.style.color = "var(--fg0)";
        }
    }

    // Event listeners for mouse actions
    const touchAvailable =
        "createTouch" in document || "ontouchstart" in window;
    if (touchAvailable) {
        canvas.addEventListener(
            "touchstart",
            (e) => {
                const event = {
                    clientX: e.changedTouches[0].clientX,
                    clientY: e.changedTouches[0].clientY,
                };
                startDrawing(event);
            },
            false,
        );
        canvas.addEventListener(
            "touchmove",
            (e) => {
                const event = {
                    clientX: e.changedTouches[0].clientX,
                    clientY: e.changedTouches[0].clientY,
                };
                draw(event);
            },
            false,
        );
        canvas.addEventListener("touchend", () => stopDrawing(pixels), false);
    } else {
        canvas.addEventListener("mousedown", startDrawing, false);
        canvas.addEventListener("mousemove", draw, false);
        canvas.addEventListener("mouseup", () => stopDrawing(pixels), false);
        canvas.addEventListener("mouseout", () => stopDrawing(pixels), false);
    }

    clearButton.addEventListener("click", () => clearAllPixels(pixels));
    eraserToggle.addEventListener("click", () => toggleMode(eraserToggle));
    floatSlider.addEventListener("input", updateBrushRadius);
    document.body.addEventListener(
        "touchmove",
        (e) => {
            if (isDrawing) {
                e.preventDefault();
            }
        },
        {
            passive: false,
        },
    );
});

/**
 * @param {string} githubRawUrl
 */
async function fetchWeights(githubRawUrl) {
    try {
        const response = await fetch(githubRawUrl);

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const jsonData = await response.json();
        return jsonData;
    } catch (error) {
        console.error("Error fetching weights:", error.message, githubRawUrl);
        return null;
    }
}
/**
 * @param {any} context
 * @param {any} canvas
 * @param {Array<number>} pixels
 */
function drawGrid(context, canvas, pixels) {
    // ensure 1:1 aspect ratio
    context.canvas.height = context.canvas.width;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const pixelSize = canvas.width / 28;

    // fill rectangles
    // context.strokeStyle = '#fff';
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            context.globalAlpha = pixels[y * 28 + x];
            const myColor = getComputedStyle(
                document.documentElement,
            ).getPropertyValue("--fg0");
            context.fillStyle = myColor;
            context.fillRect(
                x * pixelSize,
                y * pixelSize,
                pixelSize,
                pixelSize,
            );
        }
    }

    context.globalAlpha = 1;
    // create grid lines
    context.beginPath();
    for (let x = 0; x < canvas.width; x += pixelSize) {
        context.moveTo(x, 0);
        context.lineTo(x, canvas.height);
    }
    for (let y = 0; y < canvas.height; y += pixelSize) {
        context.moveTo(0, y);
        context.lineTo(canvas.width, y);
    }
    context.strokeStyle = "#ccc"; // Color of grid lines
    context.stroke();
}

/**
 * @param {Array<number>} digit
 * @param {Array<number>} w
 * @returns {number} digit @ w
 */
function vdot(digit, w) {
    if (w.length !== digit.length + 1)
        throw new Error(
            `w length ${w.length} != ${digit.length} after bias appended`,
        );
    let sum = 0;
    for (let i = 0; i < digit.length; i++) {
        sum += digit[i] * w[i];
    }
    sum += w[w.length - 1]; // bias
    return sum;
}

/**
 * @param {Array<number>} digit
 * @param {Array<Array<any>>} weights
 * @returns {Array<number>}
 */
function evalLSModel(digit, weights) {
    const scores = new Array(10).fill(0);
    for (const pairConfig of weights) {
        const [i, j, w] = pairConfig;
        const result = vdot(digit, w);
        if (result > 0) {
            scores[i] += 1;
            scores[j] -= 1;
        } else {
            scores[j] += 1;
            scores[i] -= 1;
        }
    }
    return scores;
}

/**
 * @param {any} context
 * @param {any} canvas
 * @param {boolean} drawing
 * @param {Array<number>} pixels
 * @param {number} x
 * @param {number} y
 * @param {number} r
 */
function fillPixel(context, canvas, drawing, pixels, x, y, r) {
    const ps = canvas.width / 28;
    // center of brush
    const xp = Math.floor(x / ps);
    const yp = Math.floor(y / ps);
    if (r > 0) {
        if (r <= 1.00001) {
            if (xp >= 0 && xp < 28 && yp >= 0 && yp < 28) {
                if (drawing) pixels[yp * 28 + xp] = 1;
                else pixels[yp * 28 + xp] = 0;
            }
        } else {
            const r2 = r * r;
            const rCeil = Math.ceil(r);
            for (let ry = -rCeil; ry <= rCeil; ry++) {
                for (let rx = -rCeil; rx <= rCeil; rx++) {
                    const xc = xp + rx;
                    const yc = yp + ry;
                    const dist = ry * ry + rx * rx;
                    if (dist > r2 || xc < 0 || xc >= 28 || yc < 0 || yc >= 28)
                        continue;

                    if (drawing) {
                        // guaranteed to be > 0 bc dist/r2 <= 1
                        const plateau = 0.3;
                        const alpha = Math.min(1 - dist / r2 + plateau, 1);
                        pixels[yc * 28 + xc] = Math.max(
                            pixels[yc * 28 + xc],
                            alpha,
                        );
                    } else {
                        pixels[yc * 28 + xc] = 0;
                    }
                }
            }
        }
    }
    drawGrid(context, canvas, pixels);
}

/**
 * @param {any} context
 * @param {any} canvas
 * @param {any} values
 */
function drawLogitsBarGraph(context, canvas, values) {
    const gap = 10;
    const barWidth = canvas.width / 10 - gap;
    const graphHeight = canvas.height - 30;
    const maxBarHeight = 1.0;
    const highest = Math.max(...values);
    let highestInd = -1;
    for (let i = 0; i < values.length; i++) {
        if (values[i] === highest) {
            highestInd = i;
            break;
        }
    }
    context.clearRect(0, 0, canvas.width, canvas.height);

    const style = getComputedStyle(document.documentElement);
    const fgColor = style.getPropertyValue("--fg0");
    const red = style.getPropertyValue("--red1");
    const green = style.getPropertyValue("--aqua2");
    for (let i = 0; i < values.length; i++) {
        const x = i * (barWidth + gap) + gap / 2;
        const barHeight = (Math.abs(values[i]) / maxBarHeight) * graphHeight;
        const startY = canvas.height - (barHeight + 15);
        const barColor = i === highestInd ? green : red;

        // Draw the bar
        context.fillStyle = barColor;
        context.fillRect(x, startY, barWidth, barHeight);

        context.fillStyle = fgColor;
        context.fillText(`${(values[i] * 100).toFixed()}%`, x, startY - 5);
        context.fillText(i, x + barWidth / 2, canvas.height - 5);
    }
}

/**
 * @param {any} bgContext
 * @param {any} bgCanvas
 * @param {any} barValues
 */
function drawBarGraph(bgContext, bgCanvas, barValues) {
    const gap = 10;
    const barWidth = bgCanvas.width / 10 - gap;
    const graphHeight = bgCanvas.height - 20;
    const maxBarHeight = Math.max(...barValues.map(Math.abs));
    const highest = Math.max(...barValues);
    let highestInd = -1;
    for (let i = 0; i < barValues.length; i++) {
        if (barValues[i] === highest) {
            highestInd = i;
            break;
        }
    }
    bgContext.clearRect(0, 0, bgCanvas.width, bgCanvas.height);

    const style = getComputedStyle(document.documentElement);
    const fgColor = style.getPropertyValue("--fg0");
    const red = style.getPropertyValue("--red1");
    const green = style.getPropertyValue("--aqua2");
    for (let i = 0; i < barValues.length; i++) {
        const x = i * (barWidth + gap) + gap / 2;
        const barHeight =
            (Math.abs(barValues[i]) / maxBarHeight) * (graphHeight / 2 - 15);
        const startY =
            bgCanvas.height -
            (barValues[i] < 0 ? -10 : barHeight + 10) -
            bgCanvas.height / 2;
        const barColor = i === highestInd ? green : red;

        // Draw the bar
        bgContext.fillStyle = barColor;
        bgContext.fillRect(x, startY, barWidth, barHeight);

        // Draw the bar label
        bgContext.fillStyle = fgColor;
        if (barValues[i] >= 0) {
            bgContext.fillText(barValues[i], x + barWidth / 2, startY - 5);
        } else {
            bgContext.fillText(
                barValues[i],
                x + barWidth / 2,
                startY + 10 + barHeight,
            );
        }
        bgContext.fillText(i, x + barWidth / 2, bgCanvas.height / 2);
    }
}

/**
 * @param {Array<number>} digit
 * @param {Array<any>} weights
 * @returns {Array<number>}
 */
// function evalNN(digit, weights) {
//     const digitCopy = [...digit];
//     digitCopy.push(1);
//     // layer 1 params
//     const [w1, [rows1, cols1]] = weights[0];
//     const out1 = matrixDot(
//         digitCopy,
//         w1,
//         1,
//         digitCopy.length,
//         rows1,
//         cols1,
//     ).map(relu);
//     const [w2, [rows2, cols2]] = weights[1];
//     out1.push(1);
//     const out2 = matrixDot(out1, w2, 1, out1.length, rows2, cols2);
//     return softmax(out2);
// }

function evalNN(digit, weights) {
    console.log("digit", digit, weights);
    const [
        [w1, [rows1, cols1]],
        [b1, [b1size]],
        [w2, [rows2, cols2]],
        [b2, [b2size]],
    ] = weights;
    const out1 = matrixDot(w1, digit, 128, 784, 784, 1).map(relu);
    const out2 = vectorAdd(out1, b1);
    const out3 = matrixDot(w2, out2, 10, 128, 128, 1);
    const out4 = vectorAdd(out3, b2);
    return softmax(out4);
}

function vectorAdd(v1, v2) {
    if (v1.length !== v2.length) {
        console.error(
            `unable to add vecs with lengths ${v1.length} ${v2.length}`,
        );
        return;
    }
    const out = new Array(v1.length);
    for (let i = 0; i < v1.length; i++) {
        out[i] = v1[i] + v2[i];
    }
    return out;
}

/**
 * @param {number[]} digit
 * @param {[number[], number[]][]} weights
 * @returns {number[]}
 */
function evalConv(digit, weights) {
    const [
        [f1, fshape1], // conv filter weights
        [b1, bshape1], // conv bias
        [f2, fshape2],
        [b2, fbshape2],
        [w, wshape], // fcn weights
        [b, bshape], // fcn bias
    ] = weights;

    const x1 = conv2d(1, 32, digit, 28, 28, f1, b1).map(relu);
    const x2 = maxPool2d(32, x1, 26, 26);
    const x3 = conv2d(32, 64, x2, 13, 13, f2, b2).map(relu);
    const x4 = maxPool2d(64, x3, 11, 11);
    const x5 = matrixDot(w, x4, 10, 1600, 1600, 1);
    const x6 = vsum(x5, b);
    const out = softmax(x6);
    return out;
}

/**
 * @param {number[]} arr1
 * @param {number[]} arr2
 * @returns {number[]}
 */
function vsum(arr1, arr2) {
    if (arr1.length !== arr2.length) {
        console.error("Unequal lengths for sum");
        return;
    }
    const out = new Array(arr1.length);
    for (let i = 0; i < arr1.length; i++) {
        out[i] = arr1[i] + arr2[i];
    }
    return out;
}
/**
 * @param {number[]} arr
 * @returns {number[]}
 */
function softmax(arr) {
    if (arr.length !== 10) {
        console.error(`Output of network size ${arr.length}, not 10!`);
        return;
    }
    const maxVal = Math.max(...arr);
    const expArr = arr.map((x) => Math.exp(x - maxVal));
    const expSum = expArr.reduce((sum, value) => sum + value, 0);
    return expArr.map((value) => value / expSum);
}

/**
 * @param {number[]} matrix1
 * @param {number[]} matrix2
 * @param {number} rows1
 * @param {number} cols1
 * @param {number} rows2
 * @param {number} cols2
 * @returns {number[]}
 */
function matrixDot(matrix1, matrix2, rows1, cols1, rows2, cols2) {
    // Check if the matrices can be multiplied
    if (cols1 !== rows2) {
        console.error("Invalid matrix dimensions for dot product");
        return null;
    }

    // Initialize result matrix with zeros
    const result = new Array(rows1 * cols2).fill(0);

    // Perform dot product
    for (let i = 0; i < rows1; i++) {
        for (let j = 0; j < cols2; j++) {
            for (let k = 0; k < cols1; k++) {
                result[i * cols2 + j] +=
                    matrix1[i * cols1 + k] * matrix2[k * cols2 + j];
            }
        }
    }

    return result;
}

/**
 * @param {number} x
 * @returns {number}
 */
function relu(x) {
    if (x < 0) return 0;
    return x;
}

/** Conv2d, stride 1, no padding, 3x3 kernel
 * @param {number} nInChan
 * @param {number} nOutChan
 * @param {number[]} inputData
 * @param {number} inputHeight
 * @param {number} inputWidth
 * @param {number[]} kernel
 * @param {number[]} bias
 * @returns {number[]}
 */
function conv2d(
    nInChan,
    nOutChan,
    inputData,
    inputHeight,
    inputWidth,
    kernel,
    bias,
) {
    if (inputData.length !== inputHeight * inputWidth * nInChan) {
        console.error("Invalid input size");
        return;
    }
    if (kernel.length !== 3 * 3 * nInChan * nOutChan) {
        console.error("Invalid kernel size");
        return;
    }

    const kernelHeight = 3;
    const kernelWidth = 3;

    // Compute output dimensions
    const outputHeight = inputHeight - kernelHeight + 1;
    const outputWidth = inputWidth - kernelWidth + 1;

    const output = new Array(nOutChan * outputHeight * outputWidth).fill(0);

    for (let i = 0; i < outputHeight; i++) {
        for (let j = 0; j < outputWidth; j++) {
            for (let outChan = 0; outChan < nOutChan; outChan++) {
                let sum = 0;
                // apply filter at single location over all input channels
                for (let inChan = 0; inChan < nInChan; inChan++) {
                    for (let row = 0; row < 3; row++) {
                        for (let col = 0; col < 3; col++) {
                            const inI =
                                inChan * (inputHeight * inputWidth) +
                                (i + row) * inputWidth +
                                (j + col);

                            const kI =
                                outChan * (nInChan * 3 * 3) +
                                inChan * (3 * 3) +
                                row * 3 +
                                col;
                            sum += inputData[inI] * kernel[kI];
                        }
                    }
                }
                sum += bias[outChan];
                // const idx = i * outputWidth * nOutChan + j * nOutChan + outChan;
                // add bias to sum and put in output
                const outI =
                    outChan * (outputHeight * outputWidth) +
                    i * outputWidth +
                    j;
                output[outI] = sum;
                // return;
            }
        }
    }
    return output;
}
/**
 * @param {number} nInChannels
 * @param {number[]} inputData
 * @param {number} inputHeight
 * @param {number} inputWidth
 * @returns {number[]}
 */
function maxPool2d(nInChannels, inputData, inputHeight, inputWidth) {
    if (inputData.length !== inputHeight * inputWidth * nInChannels) {
        console.error("maxpool2d: invalid input height/width");
        return;
    }
    // input shape: (26,26,32)
    // output shape:(13, 13, 32)

    // Compute output dimensions
    const poolSize = 2;
    const stride = 2;
    const outputHeight = Math.floor((inputHeight - poolSize) / stride) + 1;
    const outputWidth = Math.floor((inputWidth - poolSize) / stride) + 1;

    // Initialize output array
    const output = new Array(outputHeight * outputWidth * nInChannels).fill(0);

    // Perform max pooling
    for (let chan = 0; chan < nInChannels; chan++) {
        for (let i = 0; i < outputHeight; i++) {
            for (let j = 0; j < outputWidth; j++) {
                let m = 0;
                for (let row = 0; row < poolSize; row++) {
                    for (let col = 0; col < poolSize; col++) {
                        const ind =
                            chan * (inputHeight * inputWidth) +
                            (i * stride + row) * inputWidth +
                            (j * stride + col);
                        m = Math.max(m, inputData[ind]);
                    }
                }
                const outI =
                    chan * (outputHeight * outputWidth) + i * outputWidth + j;
                output[outI] = m;
            }
        }
    }
    return output;
}
