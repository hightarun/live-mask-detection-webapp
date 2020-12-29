import React, { useRef, useEffect, useState } from "react";
import "./style.css";
import Webcam from "react-webcam";
import maskImages from "./maskImages";
import noMaskImages from "./noMaskImages";
import * as tf from "@tensorflow/tfjs";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as knnClassifier from "@tensorflow-models/knn-classifier";

const App = () => {
  const [result, setResult] = useState("");
  const [prob, setProb] = useState("");

  const webcamRef = useRef(null);
  // const trainMaskImg = useRef(null);
  // const trainNoMaskImg = useRef(null);

  const createWebcamInput = async () => {
    //const webcamElement = React.findDOMNode(webcamRef.current.video);
    const videoElement = await webcamRef.current.video;
    const tfCam = await tf.data.webcam(videoElement);
    return tfCam;
  };

  const addImagesToDom = async () => {
    const train = document.querySelector("#trainImage");
    maskImages.map(({ id, src, alt }) => {
      const newImage = document.createElement("IMG");
      newImage.setAttribute("src", src);
      newImage.setAttribute("alt", alt);
      newImage.classList.add("mask-img");
      return train.appendChild(newImage);
    });
    noMaskImages.map(({ id, src, alt }) => {
      const newImage = document.createElement("IMG");
      newImage.setAttribute("src", src);
      newImage.setAttribute("alt", alt);
      newImage.classList.add("no-mask-img");
      return train.appendChild(newImage);
    });
    // for (let i = 0; i < 30; i++) {
    //   let src = noMaskImages[i].src;
    //   let alt = noMaskImages[i].alt;
    //   const newImage = document.createElement("IMG");
    //   newImage.setAttribute("src", src);
    //   newImage.setAttribute("alt", alt);
    //   newImage.classList.add("no-mask-img");
    //   train.appendChild(newImage);
    // }
  };

  useEffect(() => {
    const start = async () => {
      await addImagesToDom();
      console.log("IMAGES ADDED IN DOM FOR TRAINING");
      console.log("LOADING MOBILENET MODEL");
      const model = await mobilenet.load();
      console.log("MOBILENET MODEL LOADED");
      await tf.ready();
      console.log("TENSOR FLOW READY");
      const knn = knnClassifier.create();
      console.log("KNN Classifier CREATED");
      const webCamInput = await createWebcamInput();

      const trainClassifier = async () => {
        // Train using mask images
        const mask = document.querySelectorAll(".mask-img"); //const maskImages = trainMaskImg.current.childNodes;
        mask.forEach((img) => {
          try {
            const tfImg = tf.browser.fromPixels(img);
            const logits = model.infer(tfImg, "conv_preds");
            knn.addExample(logits, 0); // has mask
          } catch (error) {
            console.log(error);
          }
        });

        // Train using no mask images
        const noMask = document.querySelectorAll(".no-mask-img"); //const noMaskImages = trainNoMaskImg.current.childNodes;
        noMask.forEach((img) => {
          try {
            const tfImg = tf.browser.fromPixels(img);
            const logits = model.infer(tfImg, "conv_preds");
            knn.addExample(logits, 1); // no mask
          } catch (error) {
            console.log(error);
          }
        });
      };

      const webcamLiveDetection = async () => {
        while (true) {
          try {
            if (knn.getNumClasses() > 0) {
              const img = await webCamInput.capture();
              const activation = model.infer(img, "conv_preds");
              const result = await knn.predictClass(activation);
              img.dispose();
              setResult(result.label);
              setProb(result.confidences[result.label]);
              // Dispose the tensor to release the memory.
            }
            await tf.nextFrame();
          } catch (error) {
            console.log(error.message);
          }
        }
      };
      await trainClassifier();
      await webcamLiveDetection();
    };
    start();
  }, []);

  return (
    <div>
      <div id="trainImage"></div>
      <div className="container">
        <h1 id="head">LIVE MASK DETECTION</h1>
        <Webcam
          id="webcam"
          ref={webcamRef}
          style={{
            marginLeft: "auto",
            marginRight: "auto",
            width: 640,
            height: 480,
          }}
        />
        <div id="output">
          <h3 id="console">
            {result === "1" ? <p>NO MASK DETECTED</p> : <p>MASK DETECTED</p>}
          </h3>
          <h3 id="console">{prob}</h3>
        </div>
      </div>
    </div>
  );
};

export default App;
