/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package facerecognition;

import java.awt.AWTException;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JOptionPane;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.CvSVM;
import org.opencv.ml.CvSVMParams;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.HOGDescriptor;

/**
 *
 * @author Edo
 */
public class Interface extends javax.swing.JFrame {

    public static BufferedImage gambar;
    public static Mat matriks;
    public static boolean close = false;
    public MatOfRect matOfRect = new MatOfRect();
    private static String face_cascade_name = "haarcascade_frontalface_default.xml";
    private static String face_cascade_improved = "lbpcascade_frontalface_improved.xml";
    public static final String DIREKTORI_HAPPY = "dataset/happy/";
    public static final String DIREKTORI_SAD = "dataset/sad/";
    public static final String DIREKTORI_SHOCK = "dataset/shock/";
    public static final String DIREKTORI_ANGRY = "dataset/angry/";
    public static final String XML = "svm.xml";
    public int hitung=0;
    private static CascadeClassifier tes = new CascadeClassifier();
    boolean detectface=false;
    Rect rek = new Rect();
    public Mat pelebelan = new Mat();
    public Mat kumpulanEkstraksi = new Mat();
    public static CvSVMParams klasifikasi = new CvSVMParams();
    public HOGDescriptor d = new HOGDescriptor(new Size(196, 196), new Size(49, 49), new Size(49, 49), new Size(7, 7), 9);
    CvSVM svm = new CvSVM();
    Mat varIdx = new Mat();
    Mat samIdx = new Mat();
    /**
     * Creates new form Interface
     */
    public Interface() {
        initComponents();
    }
    
    public void settingFrame(VideoCapture webcam){
                this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		this.addWindowListener(new WindowAdapter()
		{
			@Override
			public void windowClosing(WindowEvent e)
			{
				System.out.println("Webcam ditutup");
				close=true;
				webcam.release();
				e.getWindow().dispose();
                                
       
			}
		});
    }
    
    public void streamingWebcam(Mat matframe) throws AWTException{
        
        if(detectface){
            matriks = matframe;
            tes.load(face_cascade_improved);
            Mat gray = new Mat();
            Imgproc.cvtColor(matframe, gray, Imgproc.COLOR_BGR2GRAY);
            tes.detectMultiScale(gray, matOfRect, 1.2, 2, 0, new Size(30, 30), new Size());
            Rect[] faces = matOfRect.toArray();
            for (Rect face : faces) {
                rek = face;
                Core.rectangle(matframe, face.tl(), face.br(), new Scalar(0, 255, 0, 255), 3);
                Core.putText(matframe, "DETECT FACE", new Point(face.x, face.y), Core.FONT_HERSHEY_PLAIN, 1.5, new Scalar(255, 255, 255));
            }
            
        }
        
        //Bagian ini merupakan Klasifikasi Tangan
        if(svm.get_support_vector_count()>0){
            
            //Melakukan pengubahan ukuran citra menjadi 200 x 200 pixel
            Mat Region = new Mat(matriks, rek);
            Imgproc.resize(Region, Region, new Size(200, 200));
            //Mengubah ke Grayscale
            Imgproc.cvtColor(Region, Region, Imgproc.COLOR_BGR2GRAY);
            //Mengubah tipe matriks menjadi 8UC1 dengan satu Channel
            Region.convertTo(Region, CvType.CV_8UC1);

            MatOfFloat deskritor = new MatOfFloat();
            //Melakukan perhitungan HOG
            d.compute(Region, deskritor);
            
            //Melakukan prediksi SVM
            double hasil = svm.predict(deskritor.reshape(1, 1));
            
            if(hasil==1){
                Core.putText(matframe, "HAPPY", new Point(rek.x, rek.y+150), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0, 255, 255));
            }else if(hasil==2){
                Core.putText(matframe, "SAD", new Point(rek.x, rek.y+150), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0, 255, 255));
            }else if(hasil==3){
                Core.putText(matframe, "ANGRY", new Point(rek.x, rek.y+150), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0, 255, 255));
            }else if(hasil==4){
                Core.putText(matframe, "SHOCK", new Point(rek.x, rek.y+150), Core.FONT_HERSHEY_PLAIN, 2, new Scalar(0, 255, 255));
            }
        }
        
        MatOfByte cc = new MatOfByte();
        Highgui.imencode(".JPG", matframe, cc);
        byte[] ubahByte = cc.toArray();
        InputStream ss = new ByteArrayInputStream(ubahByte);
        try {
            gambar = ImageIO.read(ss);
            Labeling.setIcon(new ImageIcon(gambar));
        } catch (IOException e) {
        }
}

    public void simpanGambar(Mat matriks, String namfile){
        switch(namfile){
            case "HAPPY":
                Imgproc.cvtColor(matriks, matriks, Imgproc.COLOR_BGR2GRAY);
                //Rect roi = new Rect(70, 50, 200, 200);
                Mat Region = new Mat(matriks, rek);
                //Region = prosesLoGSIFT(Region);
                Imgproc.resize(Region, Region, new Size(200, 200));
                hitung++;
                Highgui.imwrite(DIREKTORI_HAPPY+namfile+hitung+".jpg", Region);
                JOptionPane.showMessageDialog(rootPane, "Berhasil menyimpan gambar "+namfile+" !");
                break;
            case "SAD":
                Imgproc.cvtColor(matriks, matriks, Imgproc.COLOR_BGR2GRAY);
                //Rect roi2 = new Rect(70, 50, 200, 200);
                Mat Region2 = new Mat(matriks, rek);
                //Region2 = prosesLoGSIFT(Region2);
                Imgproc.resize(Region2, Region2, new Size(200, 200));
                hitung++;
                Highgui.imwrite(DIREKTORI_SAD+namfile+hitung+".jpg", Region2);
                JOptionPane.showMessageDialog(rootPane, "Berhasil menyimpan gambar "+namfile+" !");
                break;
            case "ANGRY":
                Imgproc.cvtColor(matriks, matriks, Imgproc.COLOR_BGR2GRAY);
                //Rect roi3 = new Rect(70, 50, 200, 200);
                Mat Region3 = new Mat(matriks, rek);
                //Region3 = prosesLoGSIFT(Region3);
                Imgproc.resize(Region3, Region3, new Size(200, 200));
                hitung++;
                Highgui.imwrite(DIREKTORI_ANGRY+namfile+hitung+".jpg", Region3);
                JOptionPane.showMessageDialog(rootPane, "Berhasil menyimpan gambar "+namfile+" !");
                break;
            case "SHOCK":
                Imgproc.cvtColor(matriks, matriks, Imgproc.COLOR_BGR2GRAY);
                //Rect roi3 = new Rect(70, 50, 200, 200);
                Mat Region4 = new Mat(matriks, rek);
                //Region3 = prosesLoGSIFT(Region3);
                Imgproc.resize(Region4, Region4, new Size(200, 200));
                hitung++;
                Highgui.imwrite(DIREKTORI_SHOCK+namfile+hitung+".jpg", Region4);
                JOptionPane.showMessageDialog(rootPane, "Berhasil menyimpan gambar "+namfile+" !");
                break;
                
        }
    }
    
    public static Mat dapatkanMat(String path) {
        Mat img = new Mat();
        Mat convert_to_gray = Highgui.imread(path, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        convert_to_gray.convertTo(img, CvType.CV_8U);
        return img;
    }
    
    public void latihHappy(){
        
        MatOfFloat nilaiDeskriptor = new MatOfFloat();
        int counts = 0;
        for(File data : new File(DIREKTORI_HAPPY).listFiles()){
            counts++;
           if(!data.equals("Thumbs.db")){
               if(!data.getAbsolutePath().contains("Thumbs.db")){
                Mat gbr = dapatkanMat(data.getAbsolutePath());
                //Proses HOG
                d.compute(gbr, nilaiDeskriptor);
                Mat labelsMat = new Mat(1, 1, CvType.CV_32SC1, new Scalar(1));
                kumpulanEkstraksi.push_back(nilaiDeskriptor.reshape(1,1));
                pelebelan.push_back(labelsMat);
               }
           }
        }  
        System.out.println(kumpulanEkstraksi);
        System.out.println(pelebelan);
        
    }
    public void latihSad(){
        
        MatOfFloat nilaiDeskriptor = new MatOfFloat();
        int counts = 0;
        for(File data : new File(DIREKTORI_SAD).listFiles()){
            counts++;
            if(!data.equals("Thumbs.db")){
                if(!data.getAbsolutePath().contains("Thumbs.db")){
                Mat gbr = dapatkanMat(data.getAbsolutePath());
                //Proses HOG
                d.compute(gbr, nilaiDeskriptor);
                Mat labelsMat = new Mat(1, 1, CvType.CV_32SC1, new Scalar(2));
                kumpulanEkstraksi.push_back(nilaiDeskriptor.reshape(1,1));
                pelebelan.push_back(labelsMat);
            }
            }
        }  
        System.out.println(kumpulanEkstraksi);
        System.out.println(pelebelan);
        
    }
    public void latihAngry(){
        
        MatOfFloat nilaiDeskriptor = new MatOfFloat();
        int counts = 0;
        for(File data : new File(DIREKTORI_ANGRY).listFiles()){
            counts++;
            if(!data.equals("Thumbs.db")){
                if(!data.getAbsolutePath().contains("Thumbs.db")){
                Mat gbr = dapatkanMat(data.getAbsolutePath());
                //Proses HOG
                d.compute(gbr, nilaiDeskriptor);
                Mat labelsMat = new Mat(1, 1, CvType.CV_32SC1, new Scalar(3));
                kumpulanEkstraksi.push_back(nilaiDeskriptor.reshape(1,1));
                pelebelan.push_back(labelsMat);
            }
            }
        }  
        System.out.println(kumpulanEkstraksi);
        System.out.println(pelebelan);
        
    }
    public void latihShock(){
        
        MatOfFloat nilaiDeskriptor = new MatOfFloat();
        int counts = 0;
        for(File data : new File(DIREKTORI_SHOCK).listFiles()){
            counts++;
            if(!data.equals("Thumbs.db")){
                if(!data.getAbsolutePath().contains("Thumbs.db")){
                Mat gbr = dapatkanMat(data.getAbsolutePath());
                //Proses HOG
                d.compute(gbr, nilaiDeskriptor);
                Mat labelsMat = new Mat(1, 1, CvType.CV_32SC1, new Scalar(4));
                kumpulanEkstraksi.push_back(nilaiDeskriptor.reshape(1,1));
                pelebelan.push_back(labelsMat);
            }
            }
        }  
        System.out.println(kumpulanEkstraksi);
        System.out.println(pelebelan);
        
    }
    
    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        Labeling = new javax.swing.JLabel();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jButton3 = new javax.swing.JButton();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jButton1.setText("TAKE FOR DATASET");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("DETECT FACE");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jButton3.setText("PREDICT EMOTION");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(Labeling, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
            .addGroup(layout.createSequentialGroup()
                .addGap(146, 146, 146)
                .addComponent(jButton1)
                .addGap(18, 18, 18)
                .addComponent(jButton2)
                .addGap(18, 18, 18)
                .addComponent(jButton3)
                .addContainerGap(162, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(Labeling, javax.swing.GroupLayout.PREFERRED_SIZE, 357, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(30, 30, 30)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButton1)
                    .addComponent(jButton2)
                    .addComponent(jButton3))
                .addContainerGap(44, Short.MAX_VALUE))
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        // TODO add your handling code here:
        
        detectface = true;
        
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed
        // TODO add your handling code here:
        String kind = JOptionPane.showInputDialog(rootPane, "What expression will you train ? (Type HAPPY/SHOCK/SAD/ANGRY)");
        simpanGambar(matriks, kind);
        
    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        // TODO add your handling code here:
        
        latihHappy();
        latihSad();
        latihAngry();
        latihShock();
        
        Mat fixdata = new Mat();
        Mat fixlabel = new Mat();
        
        kumpulanEkstraksi.copyTo(fixdata);
        fixdata.convertTo(fixdata, CvType.CV_32F);
        pelebelan.copyTo(fixlabel);
        fixlabel.convertTo(fixlabel, CvType.CV_32S);
        
        //Proses Melatih Data Latih Menggunakan SVM
        klasifikasi.set_svm_type(CvSVM.C_SVC);
        klasifikasi.set_kernel_type(CvSVM.POLY);
        klasifikasi.set_degree(0.5);
        klasifikasi.set_gamma(1);
        klasifikasi.set_coef0(0);
        klasifikasi.set_C(7);
        klasifikasi.set_nu(0.5);
        klasifikasi.set_p(0.0);
        klasifikasi.set_term_crit(new TermCriteria(TermCriteria.MAX_ITER, (int) 1e7, 1e-6));
        
        //HOG
        svm.train(kumpulanEkstraksi, pelebelan, varIdx, samIdx, klasifikasi);
        svm.save(XML);
        svm.load(XML);
        JOptionPane.showMessageDialog(rootPane, "Sukses Melatih Dataset!");
        
        
    }//GEN-LAST:event_jButton3ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(Interface.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new Interface().setVisible(true);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JLabel Labeling;
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    // End of variables declaration//GEN-END:variables
}
