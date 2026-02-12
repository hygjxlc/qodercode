package com.example.expert.config;

import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.Contact;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class OpenApiConfig {
    
    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("会务专家管理系统 API")
                        .version("1.0.0")
                        .description("会务专家信息管理系统后端接口文档")
                        .contact(new Contact()
                                .name("技术支持")
                                .email("support@example.com")));
    }
}
