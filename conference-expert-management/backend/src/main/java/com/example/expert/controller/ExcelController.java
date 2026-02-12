package com.example.expert.controller;

import com.example.expert.dto.ExpertDTO;
import com.example.expert.service.ExcelService;
import com.example.expert.service.ExpertService;
import lombok.RequiredArgsConstructor;
import org.springframework.core.io.InputStreamResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/excel")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class ExcelController {
    
    private final ExcelService excelService;
    private final ExpertService expertService;
    
    @GetMapping("/export")
    public ResponseEntity<?> exportExperts() throws IOException {
        List<ExpertDTO> experts = expertService.getAllExperts(org.springframework.data.domain.Pageable.unpaged()).getContent();
        ByteArrayInputStream stream = excelService.exportExperts(experts);
        
        InputStreamResource resource = new InputStreamResource(stream);
        
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=experts.xlsx")
                .contentType(MediaType.parseMediaType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                .body(resource);
    }
    
    @GetMapping("/template")
    public ResponseEntity<?> downloadTemplate() throws IOException {
        ByteArrayInputStream stream = excelService.getImportTemplate();
        
        InputStreamResource resource = new InputStreamResource(stream);
        
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=expert_import_template.xlsx")
                .contentType(MediaType.parseMediaType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                .body(resource);
    }
    
    @PostMapping("/import")
    public ResponseEntity<?> importExperts(@RequestParam("file") MultipartFile file) {
        try {
            List<ExpertDTO> imported = excelService.importExperts(file);
            
            // Save imported experts
            int successCount = 0;
            for (ExpertDTO dto : imported) {
                try {
                    expertService.createExpert(dto);
                    successCount++;
                } catch (Exception e) {
                    // Continue with next
                }
            }
            
            Map<String, Object> response = new HashMap<>();
            response.put("success", true);
            response.put("message", "成功导入 " + successCount + " 条记录");
            response.put("total", imported.size());
            return ResponseEntity.ok(response);
        } catch (RuntimeException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", e.getMessage());
            return ResponseEntity.badRequest().body(error);
        } catch (IOException e) {
            Map<String, Object> error = new HashMap<>();
            error.put("success", false);
            error.put("message", "文件读取失败: " + e.getMessage());
            return ResponseEntity.badRequest().body(error);
        }
    }
}
