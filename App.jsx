// import { useState } from 'react'
// import upload from '../assets/upload-img.png'
// import '../styles/App.css'
// import tickImage from '../assets/tick.png'; // Import the tick image

// function App() {
//   const [selectedFiles, setSelectedFiles] = useState([]); // Store an array of files
//   const [uploaded, setUploaded] = useState(false); // Add a new state to track if a file has been uploaded

//   const handleUploadClick = async () => {
//     if (window.showOpenFilePicker) {
//       try {
//         const fileHandles = await window.showOpenFilePicker({
//           types: [
//             {
//               description: 'Images',
//               accept: {
//                 'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
//               },
//             },
//           ],
//           excludeAcceptAllOption: true,
//           multiple: true, // Allow multiple file selection
//         });
//         if (fileHandles.length > 0) {
//           const files = await Promise.all(fileHandles.map((fileHandle) => fileHandle.getFile()));
//           setSelectedFiles((prevFiles) => [...prevFiles, ...files]); // Append new files to the existing array
//           setUploaded(true); // Set uploaded to true when a file is uploaded
//         }
//       } catch (error) {
//         console.error(error);
//       }
//     } else {
//       console.log('showOpenFilePicker is not supported in this browser');
//     }
//   };

//   return (
//     <div className="app">
//       <div className="title">
//         <p>Project Name</p>
//         <p>This is line 2</p>
//       </div>

//       <div className="upload_div" onClick={handleUploadClick}>
//         <p>Upload Image</p>
//         <div className="up_img" style={{
//           backgroundImage: uploaded ? 'none' : `url(${upload})`
//         }}>
//           {uploaded ? (
//             <img src={tickImage} alt="Uploaded successfully" />
//           ) : (
//             <div></div>
//           )}
//         </div>
//       </div>

//       {selectedFiles.length > 0 && (
//         <div className="imageField">
//           <div className="ip-images">
//             <h3>Uploaded Images</h3>
//             <div 
//               className="ips" 
//               style={{
//                 display: 'flex',
//                 flexWrap: 'wrap',
//                 justifyContent: 'space-between'
//               }}
//             >
//               {selectedFiles.map((file, index) => (
//                 <div 
//                   key={index} 
//                   style={{
//                     width: 'calc(384px + 2%)', // Add 2% to the width for the gap
//                     marginRight: index === selectedFiles.length - 1 ? 0 : '2%' // Remove margin for the last image
//                   }}
//                 >
//                   <img 
//                     src={URL.createObjectURL(file)} 
//                     alt={file.name} 
//                     style={{
//                       width: '100%',
//                       height: '512px',
//                       objectFit: 'cover'
//                     }}
//                   />
//                 </div>
//               ))}
//             </div>
//           </div>

//           <div 
//             className="op-images" 
//             style={{
//               width: '50%',
//               height: '50%',
//               display: 'flex',
//               justifyContent: 'center',
//               alignItems: 'center'
//             }}
//           >
//             {/* <!-- Output images will be displayed here --> */}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default App;





// import { useState } from 'react'
// import upload from '../assets/upload-img.png'
// import '../styles/App.css'
// import tickImage from '../assets/tick.png'; // Import the tick image

// function App() {
//   const [selectedFiles, setSelectedFiles] = useState([]); // Store an array of files
//   const [uploaded, setUploaded] = useState(false); // Add a new state to track if a file has been uploaded

//   const handleUploadClick = async () => {
//     if (window.showOpenFilePicker) {
//       try {
//         const fileHandles = await window.showOpenFilePicker({
//           types: [
//             {
//               description: 'Images',
//               accept: {
//                 'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
//               },
//             },
//           ],
//           excludeAcceptAllOption: true,
//           multiple: true, // Allow multiple file selection
//         });
//         if (fileHandles.length > 0) {
//           const files = await Promise.all(fileHandles.map((fileHandle) => fileHandle.getFile()));
//           setSelectedFiles((prevFiles) => [...prevFiles, ...files]); // Append new files to the existing array
//           setUploaded(true); // Set uploaded to true when a file is uploaded
//         }
//       } catch (error) {
//         console.error(error);
//       }
//     } else {
//       console.log('showOpenFilePicker is not supported in this browser');
//     }
//   };

//   return (
//     <div className="app">

//       <div className="title">
//         <p className='prjct-name'>Project Name</p>
//       </div>

//       <div className="team">

//       </div>

//       <div className="upload_div" onClick={handleUploadClick}>
//         <p>Upload Image</p>
//         <div className="up_img" style={{
//           backgroundImage: uploaded ? 'none' : `url(${upload})`
//         }}>
//           {uploaded ? (
//             <img src={tickImage} alt="Uploaded successfully" />
//           ) : (
//             <div></div>
//           )}
//         </div>
//       </div>

//       {selectedFiles.length > 0 && (
//         <div className="imageField">
//           <div className="ip-images">
//             <h3>Uploaded Images</h3>
//             <div
//               className="ips"
//               style={{
//                 display: 'flex',
//                 flexWrap: 'wrap',
//                 justifyContent: 'space-between'
//               }}
//             >
//               {selectedFiles.map((file, index) => (
//                 <div
//                   key={index}
//                   style={{
//                     width: 'calc(384px + 2%)', // Add 2% to the width for the gap
//                     marginRight: index === selectedFiles.length - 1 ? 0 : '2%' // Remove margin for the last image
//                   }}
//                 >
//                   <img
//                     src={URL.createObjectURL(file)}
//                     alt={file.name}
//                     style={{
//                       width: '100%',
//                       height: '512px',
//                       objectFit: 'cover'
//                     }}
//                   />
//                 </div>
//               ))}
//             </div>
//           </div>

//           <div
//             className="op-images"
//             style={{
//               width: '50%',
//               height: '50%',
//               display: 'flex',
//               justifyContent: 'center',
//               alignItems: 'center'
//             }}
//           >
//             {/* <!-- Output images will be displayed here --> */}
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default App;




import { useState } from 'react'
import upload from '../assets/upload-img.png'
import '../styles/App.css'
import tickImage from '../assets/tick.png'; // Import the tick image

function App() {
  const [selectedFiles, setSelectedFiles] = useState([]); // Store an array of files
  const [uploaded, setUploaded] = useState(false); // Add a new state to track if a file has been uploaded

  const handleUploadClick = async () => {
    if (window.showOpenFilePicker) {
      try {
        const fileHandles = await window.showOpenFilePicker({
          types: [
            {
              description: 'Images',
              accept: {
                'image/*': ['.png', '.jpg', '.jpeg', '.gif'],
              },
            },
          ],
          excludeAcceptAllOption: true,
          multiple: true, // Allow multiple file selection
        });
        if (fileHandles.length > 0) {
          const files = await Promise.all(fileHandles.map((fileHandle) => fileHandle.getFile()));
          setSelectedFiles((prevFiles) => [...prevFiles, ...files]); // Append new files to the existing array
          setUploaded(true); // Set uploaded to true when a file is uploaded
        }
      } catch (error) {
        console.error(error);
      }
    } else {
      console.log('showOpenFilePicker is not supported in this browser');
    }
  };

  return (
    <div className="app">

      <div className="title">
        <p className='prjct-name'>Shadow Quest</p>
      </div>


      <div className="about">
        <p>This project aims to enhance (Low Light Image Enhancement) the feeble light reflected from PSR regions of Lunar craters into a better SNR image for interpretations. Challenge: Feeble signal to better signal image generation. Low light image noise removal. Usage: For generating first of its kind PSR image map of lunar poles captured by OHRC of Chandrayaan-2. Users: Landing site selection users and geomorphological application users. Available Solutions (if Yes, reasons for not using them): Specific solution of Chandrayaan-2 needs to be developed. General techniques and algorithms are available. Desired Outcome: Software for generating low light image enhancement.</p>
      </div>
    </div>
  );
};

export default App;