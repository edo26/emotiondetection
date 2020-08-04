/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facerecognition;

import java.awt.AWTException;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.VideoCapture;

/**
 *
 * @author Edo
 */
public class FaceRecognition {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws AWTException {
        // TODO code application logic here
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        VideoCapture webcam = new VideoCapture(0);
        Mat matriks = new Mat();
        Interface frame = new Interface();
        frame.settingFrame(webcam);
        frame.setVisible(true);
        while (true && !Interface.close) {
            if (!webcam.isOpened()) {
                JOptionPane.showMessageDialog(null, "Ada masalah pada web-cam anda !.");
            } else {
                if (webcam.read(matriks)) {
                    Core.flip(matriks, matriks, 1);
                    frame.streamingWebcam(matriks);
                } else {
                    JOptionPane.showMessageDialog(null, "Ada yang salah !.");
                }

            }
        }
    }
    
}
