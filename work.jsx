// Work.jsx
// import React from 'react';
// import upload from '../assets/upload-img.png';
// import tickImage from '../assets/tick.png'; // Import the tick image
// import "../styles/work.css"


// function Work() {
//   const [selectedFiles, setSelectedFiles] = React.useState([]); // Store an array of files
//   const [uploaded, setUploaded] = React.useState(false); // Add a new state to track if a file has been uploaded

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
//     <div className='Work'>
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
// }

// export default Work;


import React from 'react';
import upload from '../assets/upload-img.png';
import tickImage from '../assets/tick.png'; // Import the tick image
import "../styles/work.css"

function Work() {
  const [selectedFiles, setSelectedFiles] = React.useState([]); // Store an array of files
  const [uploaded, setUploaded] = React.useState(false); // Add a new state to track if a file has been uploaded

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
    <div className='Work'>
      <div className="upload_div" onClick={handleUploadClick}>
        <p>Upload Image</p>
        <div className="up_img" style={{
          backgroundImage: uploaded ? 'none' : `url(${upload})`
        }}>
          {uploaded ? (
            <img src={tickImage} alt="Uploaded successfully" />
          ) : (
            <div></div>
          )}
        </div>
      </div>

      {selectedFiles.length > 0 && (
        <div className="imageField">
          <div className="ip-images">
            <h3>Uploaded Images</h3>
            <div
              className="ips"
              style={{
                display: 'flex',
                flexWrap: 'wrap',
                justifyContent: 'space-between'
              }}
            >
              {selectedFiles.map((file, index) => (
                <div
                  key={index}
                  style={{
                    width: 'calc(300px + 2%)', // Add 2% to the width for the gap
                    marginRight: index === selectedFiles.length - 1 ? 0 : '2%' // Remove margin for the last image
                  }}
                >
                  <img
                    src={URL.createObjectURL(file)}
                    alt={file.name}
                    className="uploaded-image"
                  />
                </div>
              ))}
            </div>
          </div>

          <div
            className="op-images"
            style={{
              width: '50%',
              height: '50%',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            {/* <!-- Output images will be displayed here --> */}
          </div>
        </div>
      )}
    </div>
  );
}

export default Work;



